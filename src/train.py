import os
import torch
import torch.nn.functional as F
from src.model import Seq2Seq
from torch.utils.data import DataLoader
from src.eval import accuracy, ModelEvaluator
from torch.optim import Optimizer
from tqdm import tqdm
from src.tokenizer import BPETokenizer
from src.processing import EnglishSpanishDataset, BatchProcess
from src.config import Config

CHECKPOINT_FILE = "en2es_checkpoint.pth"

def prepare_loaders(tokenizer: BPETokenizer, cfg: Config):
	batch_process = BatchProcess(pad_idx=tokenizer.pad_idx)
	generator = torch.Generator().manual_seed(99)
	train_dataset = EnglishSpanishDataset(tokenizer, generator, split="train")
	test_dataset = EnglishSpanishDataset(tokenizer, generator, split="test")
	train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=batch_process, generator=generator)
	test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=batch_process, generator=generator)
	return train_loader, test_loader

class Trainer:
	def __init__(self, 
			  	 model: Seq2Seq,
				 tokenizer: BPETokenizer,
				 train_loader: DataLoader,
				 test_loader: DataLoader,
				 device: str,
				 pad_idx: int,
				 log_step: int = 100,
				 ):


		self.generator = torch.Generator().manual_seed(99)
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.device = device
		self.pad_idx = pad_idx
		self.log_step = log_step
		self.checkpoint_file = CHECKPOINT_FILE

		self.eval = ModelEvaluator(
			model,
			device,
			test_loader,
			tokenizer
		)

		self.optimizer = None
		self.checkpoint_data = None

		# train history
		self.train_losses = []
		self.train_accs = []

		# test_history
		self.test_losses = []
		self.test_accs = []
		self.bleu_scores = []

	def save_checkpoint(self, optimizer: Optimizer, epoch: int, total_epochs: int, loss: float):
		state_dict = {
			"optimizer_state_dict": optimizer.state_dict(),
			"model_state_dict": self.model.state_dict(),
			"epoch": epoch,
			"loss": loss,
			"config": self.model.config,
			"num_epochs": total_epochs,
		}
		torch.save(state_dict, self.checkpoint_file)

	@classmethod
	def from_config(cls, cfg: Config) -> 'Trainer':
		device = "cuda" if torch.cuda.is_available() else "cpu"
		tokenizer = BPETokenizer.load("en_es.json")
		train_loader, test_loader = prepare_loaders(tokenizer, cfg)
		model = Seq2Seq(cfg).to(device)

		trainer = Trainer(
			model,
			tokenizer,
			train_loader,
			test_loader,
			device,
			pad_idx=tokenizer.pad_idx,
			log_step=50
		)
		return trainer

	@classmethod
	def from_checkpoint(cls, path: str = CHECKPOINT_FILE) -> 'Trainer':
		if not os.path.exists(path):
			raise FileNotFoundError(f"Checkpoint file not found: {path}")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		data = torch.load(path, map_location=device, weights_only=False)
		tokenizer = BPETokenizer.load("en_es.json")
		cfg: Config = data["config"]
		train_loader, test_loader = prepare_loaders(tokenizer, cfg)

		model = Seq2Seq(cfg).to(device)
		model.load_state_dict(data["model_state_dict"])

		trainer = cls(model, tokenizer, train_loader, test_loader, device, pad_idx=tokenizer.pad_idx)
		trainer.checkpoint_data = data
		trainer.optimizer = torch.optim.AdamW(model.parameters())
		trainer.optimizer.load_state_dict(data["optimizer_state_dict"])
		return trainer

	def resume(self):
		if self.checkpoint_data is None:
			raise RuntimeError("No checkpoint loaded. Use Trainer.from_checkpoint() to resume training.")
		start_epoch = self.checkpoint_data["epoch"] + 1
		num_epochs = self.checkpoint_data["num_epochs"]
		return self._train(self.optimizer, start_epoch=start_epoch, num_epochs=num_epochs)
		
	def _train(self, optimizer: Optimizer, start_epoch:int = 0, num_epochs: int = 10):
		self.model.train()
		epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Epochs")
		for epoch in epoch_bar:
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
					self.save_checkpoint(
						optimizer,
						epoch,
						num_epochs,
						avg_loss
					)

		self.train_accs = torch.stack(self.train_accs)
		self.test_accs = torch.stack(self.test_accs)

		return self.train_losses, self.train_accs, self.test_losses, self.test_accs, self.bleu_scores

	def train(self, lr: float, epochs: int = 1):
		optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
		return self._train(optimizer, start_epoch=0, num_epochs=epochs)