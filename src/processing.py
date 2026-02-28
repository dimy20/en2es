import torch
from torch.utils.data import Dataset, random_split
from src.tokenizer import BPETokenizer
from datasets import load_dataset

class BatchProcess:
	def __init__(self, pad_idx):
		self.pad_idx = pad_idx

	def __call__(self, batch):
		# Adds padding to both target and source sententes.
		X, Y = [], []
		max_x, max_y = 0, 0

		for sample in batch:
			x, y = sample
			max_x = max(len(x), max_x)
			max_y = max(len(y), max_y)

		for sample in batch:
			x, y = sample

			padded_x = torch.cat([x, torch.tensor([self.pad_idx] * (max_x - len(x)), dtype=torch.long)])
			padded_y = torch.cat([y, torch.tensor([self.pad_idx] * (max_y - len(y)), dtype=torch.long)])

			X.append(padded_x)
			Y.append(padded_y)

		return torch.vstack(X), torch.vstack(Y)

class EnglishSpanishDataset(Dataset):
	def __init__(self, tokenizer: BPETokenizer, generator: torch.Generator, split: str):
		super().__init__()
		self.ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=split)
		self.tokenizer = tokenizer
		self.generator = generator

	def __getitem__(self, index): 
		t = self.ds[index].get("translation")
		en = t.get("en")
		es = t.get("es")
		return self.tokenizer.encode(en), self.tokenizer.encode(es)

	def __len__(self):
		return len(self.ds)
		#return len(self.df[self.tokenizer.target])
	
	def split(self):
		train_dataset, test_dataset = random_split(self, [0.80, 0.20], generator=self.generator)
		return train_dataset, test_dataset
	