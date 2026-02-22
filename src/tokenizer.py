import torch
import re
import pandas as pd
import json
import nltk
from pathlib import Path
nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize
TARGETS = ["english", "spanish"]
SAVE_PATH = "./data"

class Tokenizer:
	def __init__(self, target: str, df: pd.DataFrame = None):
		if target and target not in TARGETS:
			raise ValueError(f"Target must be one of: {TARGETS}")

		self.target = target
		self.df = df

		# special tokens
		self.eos_token = "<EOS>" # end of sequence
		self.sos_token = "<SOS>" # start of sequence
		self.unk_token = "<UNK>" # unknown 
		self.pad_token = "<PAD>" # padding 

		self.special_tokens = [
			self.pad_token,
			self.eos_token,
			self.sos_token,
			self.unk_token,
		]

		self.vocab_size = 0
		self.token_to_idx = {}
		self.idx_to_token = {}

	def init_mappings(self, vocab: list[str]):
		self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
		self.idx_to_token = {idx: token for idx, token in enumerate(vocab)}
		self.vocab_size = len(vocab)

	def tokenize(self, s: str) -> list[str]:
		return word_tokenize(s)
		#s = re.sub(r"([.!?])", r" \1", s)
		#tokens = s.split()
		#return tokens

	def build_vocab(self):
		vocab = set()
		sentences = self.df[self.target]
		for s in sentences:
			tokens = self.tokenize(s)
			for token in tokens:
				vocab.add(token)

		vocab = self.special_tokens + sorted(list(vocab))
		self.init_mappings(vocab)
		self.save(vocab)

	@staticmethod
	def from_pretrained(target: str) -> 'Tokenizer':
		fname = Path(SAVE_PATH) / f"tokens_{target}.json"
		tokenizer = Tokenizer(target=target)
		with open(fname, "r") as f:
			data = json.loads(f.read())
			vocab = data.get("vocab", [])
			if len(vocab) == 0:
				raise ValueError(f"Vocabulary empty or not found in {fname}")

			tokenizer.init_mappings(vocab)
			tokenizer.target = data.get("target")
		
		return tokenizer

	@property
	def fname(self):
		return Path(SAVE_PATH) / f"tokens_{self.target}.json"

	def save(self, vocab: list[str]):
		data = {
			"target": self.target,
			"vocab": vocab
		}
		
		with open(self.fname, "w") as f:
			f.write(json.dumps(data, indent=4))

	def encode(self, s: str) -> torch.Tensor:
		SOS = [self.token_to_idx[self.sos_token]]
		EOS = [self.token_to_idx[self.eos_token]]
		out = []
		for token in self.tokenize(s):
			idx = self.token_to_idx.get(token, self.token_to_idx[self.unk_token])
			out.append(idx)
		return torch.tensor(SOS + out + EOS)

	def decode(self, indices: list[int]) -> str:
		out = []
		UNK_IDX = self.token_to_idx[self.unk_token]
		EOS_IDX = self.token_to_idx[self.eos_token]
		SOS_IDX = self.token_to_idx[self.sos_token]
		for idx in indices:
			if idx == SOS_IDX:
				continue
			if idx == EOS_IDX:
				break
			out.append(self.unk_token if idx == UNK_IDX else self.idx_to_token[idx])
		return " ".join(out)

	
