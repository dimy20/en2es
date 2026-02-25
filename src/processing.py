import torch
from torch.utils.data import Dataset, random_split
from src.tokenizer import Tokenizer
import pandas as pd
from pathlib import Path
from typing import Tuple
DATASET = "data/data.csv"

class BatchProcess:
	def __init__(self, pad_idx_src, pad_idx_dst):
		self.pad_idx_src = pad_idx_src
		self.pad_idx_dst = pad_idx_dst

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

			padded_x = torch.cat([x, torch.tensor([self.pad_idx_src] * (max_x - len(x)), dtype=torch.long)])
			padded_y = torch.cat([y, torch.tensor([self.pad_idx_dst] * (max_y - len(y)), dtype=torch.long)])

			X.append(padded_x)
			Y.append(padded_y)

		return torch.vstack(X), torch.vstack(Y)

class EnglishSpanishDataset(Dataset):
	def __init__(self, en_tokenizer: Tokenizer, es_tokenizer: Tokenizer, generator: torch.Generator):
		super().__init__()
		self.df = pd.read_csv(Path.absolute(Path(DATASET)))
		self.en_tokenizer = en_tokenizer
		self.es_tokenizer = es_tokenizer
		self.generator = generator

		assert len(self.df[self.en_tokenizer.target]) == len(self.df[self.es_tokenizer.target])

	def __getitem__(self, index): 
		en = self.df[self.en_tokenizer.target]
		es = self.df[self.es_tokenizer.target]

		src = en[index]
		target = es[index]
		return self.en_tokenizer.encode(src), self.es_tokenizer.encode(target)

	def __len__(self):
		return len(self.df[self.en_tokenizer.target])
	
	def split(self):
		train_dataset, test_dataset = random_split(self, [0.80, 0.20], generator=self.generator)
		return train_dataset, test_dataset
	