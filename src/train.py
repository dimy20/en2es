import torch
import torch.nn.functional as F
from src.model import Seq2Seq
from torch.utils.data import DataLoader

# Accuracy on one batch on data.
# Accuracy is not the best metric for machine translation, im going to include it anyways
# need to add BLEU score.
def accuracy(pad_idx: int, Y: torch.Tensor, y_log_probs: torch.Tensor):
	# i dont want to include pad indices in the accuracy
	indices = Y[:, 1:] != pad_idx
	yreal = Y[:, 1:][indices]  # real indices
	ypredicted = torch.argmax(y_log_probs[:, :, :], dim=-1)[indices] # predicted indices
	return torch.mean((yreal == ypredicted).float(), dtype=torch.float)

def eval(model: Seq2Seq, 
		 test_loader: DataLoader,
		 pad_idx: int,
		 device: str
		 ):
	test_loss = 0.0
	test_acc = 0.0
	model.eval()
	with torch.no_grad():
		test_loss = 0

		for Xt, Yt in test_loader:
			Xt = Xt.to(device)
			Yt = Yt.to(device)

			# forward pass on test batch
			y_logits_t = model.forward(Xt, Yt)
			y_log_probs_t = F.log_softmax(y_logits_t, dim=-1)
			B, T, Y_VOCAB = y_log_probs_t.shape
			Lt = F.nll_loss(y_log_probs_t.view(B*T, Y_VOCAB), 
						Yt[:, 1:].contiguous().view(-1),
						ignore_index=pad_idx)

			test_loss += Lt.item()
			test_acc += accuracy(pad_idx, Y=Yt, y_log_probs=y_log_probs_t)

	model.train()

	return test_loss / len(test_loader), test_acc / len(test_loader)

class Trainer:
	def __init__(self, 
			  	 model: Seq2Seq,
				 train_loader: DataLoader,
				 test_loader: DataLoader,
				 device: str,
				 pad_idx: int,
				 log_step: int = 100
				 ):

		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.device = device
		self.pad_idx = pad_idx
		self.log_step = log_step

		# train history
		self.train_losses = []
		self.train_accs = []

		# test_history
		self.test_losses = []
		self.test_accs = []

	def train(self, lr: float, epochs: int = 1):
		self.model.train()
		optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0)

		for _ in range(epochs):
			loss = 0.0
			train_acc = 0.0
			for batch_idx, (X, Y) in enumerate(self.train_loader):
				X = X.to(self.device)
				Y = Y.to(self.device)

				optimizer.zero_grad()

				y_logits = self.model.forward(X, Y)
				y_log_probs = F.log_softmax(y_logits, dim=-1)
				B, T, Y_VOCAB = y_log_probs.shape
				#L = F.nll_loss(y_log_probs.view(B*T, Y_VOCAB), Y.view(B * T)) # SKIP <SOS>

				# Batched version of nll_loss call, this function doesnt like batches so we need to adjust the shapes.
				L = F.nll_loss(y_log_probs.view(B*T, Y_VOCAB), 
							Y[:, 1:].contiguous().view(-1),
							#Y[:, 1:].reshape(B * (Y.size(1) - 1)), 
							ignore_index=self.pad_idx)


				train_acc += accuracy(pad_idx=self.pad_idx, 
									Y=Y, 
									y_log_probs=y_log_probs)

				loss += L.item()

				#backprop
				L.backward()

				#clip gradients
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

				#update params
				optimizer.step()
				
				#record loss
				if (batch_idx + 1) % self.log_step == 0:
					self.train_losses.append(loss / self.log_step)
					self.train_accs.append(train_acc / self.log_step)

					loss = 0
					train_acc = 0 

					# run loss and acc on test set
					test_loss, test_acc = eval(self.model, self.test_loader, self.pad_idx, self.device)

					self.test_losses.append(test_loss)
					self.test_accs.append(test_acc)

		self.train_accs = torch.stack(self.train_accs)
		self.test_accs = torch.stack(self.test_accs)

		return self.train_losses, self.train_accs, self.test_losses, self.test_accs