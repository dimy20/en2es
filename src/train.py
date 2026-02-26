import torch
import torch.nn.functional as F
from src.model import Seq2Seq
from torch.utils.data import DataLoader
from src.eval import accuracy, ModelEvaluator
from tqdm import tqdm

class Trainer:
	def __init__(self, 
			  	 model: Seq2Seq,
				 train_loader: DataLoader,
				 test_loader: DataLoader,
				 device: str,
				 pad_idx: int,
				 eval: ModelEvaluator,
				 log_step: int = 100,
				 ):

		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.device = device
		self.pad_idx = pad_idx
		self.log_step = log_step
		self.eval = eval
		#

		# train history
		self.train_losses = []
		self.train_accs = []

		# test_history
		self.test_losses = []
		self.test_accs = []
		self.bleu_scores = []

	def train(self, lr: float, epochs: int = 1):
		self.model.train()
		optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

		epoch_bar = tqdm(range(epochs), desc="Epochs")
		for _ in epoch_bar:
			loss = 0.0
			train_acc = 0.0
			batch_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Batches", leave=False)
			for batch_idx, (X, Y) in batch_bar:
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

				batch_bar.set_postfix(loss=f"{L.item():.4f}")
				
				#record loss
				if (batch_idx + 1) % self.log_step == 0:
					avg_loss = loss / self.log_step
					avg_acc = train_acc / self.log_step

					self.train_losses.append(avg_loss)
					self.train_accs.append(avg_acc)

					loss = 0
					train_acc = 0 

					# run loss and acc on test set
					test_loss, test_acc, bleu_score = self.eval.eval_test()
					#test_loss, test_acc = eval_test_set(self.model, self.test_loader, self.pad_idx, self.device)

					self.test_losses.append(test_loss)
					self.test_accs.append(test_acc)
					self.bleu_scores.append(bleu_score)

					epoch_bar.set_postfix(
						train_loss=f"{avg_loss:.4f}",
						train_acc=f"{avg_acc:.4f}",
						test_loss=f"{test_loss:.4f}",
						test_acc=f"{test_acc:.4f}",
						bleu=f"{bleu_score:.2f}"
					)

		self.train_accs = torch.stack(self.train_accs)
		self.test_accs = torch.stack(self.test_accs)

		return self.train_losses, self.train_accs, self.test_losses, self.test_accs, self.bleu_scores