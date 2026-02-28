import torch
import pandas as pd
import json
import nltk
from pathlib import Path
import os
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import DatasetDict
import warnings

TARGETS = ["english", "spanish"]
VOCAB_PATH = "./data/vocabs"
MAX_SEQ_LEN = 256

class WordTokenizer:
	def __init__(self, target: str, df: pd.DataFrame = None):
		warnings.warn(
            "WordTokenizer is deprecated, BPETokenizer is used as the tokenizer now.",
            DeprecationWarning,
            stacklevel=2
        )
		nltk.download("punkt_tab")
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
		try:
			return word_tokenize(s)
		except TypeError as e:
			print(f"Failed to tokenize {s}, Error: {e}")
			return []

	def build_vocab(self):
		vocab = set()
		sentences = self.df[self.target]
		
		for s in tqdm(sentences, desc=f"Building vocab ({self.target})", unit="sent"):
			tokens = self.tokenize(s)
			for token in tokens:
				vocab.add(token)

		vocab = self.special_tokens + sorted(list(vocab))
		self.init_mappings(vocab)
		self.save(vocab)

	@staticmethod
	def from_pretrained(target: str) -> 'Tokenizer':
		fname = Path(VOCAB_PATH) / f"tokens_{target}.json"
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
	def sos_idx(self):
		return self.token_to_idx[self.sos_token]

	@property
	def eos_idx(self):
		return self.token_to_idx[self.eos_token]

	@property
	def pad_idx(self):
		return self.token_to_idx[self.pad_token]

	@property
	def fname(self):
		return Path(VOCAB_PATH) / f"tokens_{self.target}.json"

	def save(self, vocab: list[str]):
		data = {
			"target": self.target,
			"vocab": vocab
		}

		path = Path(VOCAB_PATH)
		if not os.path.exists(path):
			os.makedirs(path)

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

	def decode_batch(self, indices: torch.Tensor) -> list[str]:
		return [self.decode(seq.tolist()) for seq in indices]

class BPETokenizer:
	def __init__(self, vocab_size: int = 16000):
		self.vocab_size = vocab_size
		self.tokenizer = None

	def train(self, dataset: DatasetDict):
		tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
		tokenizer.pre_tokenizer = Whitespace()

		trainer = BpeTrainer(
			vocab_size=self.vocab_size,
			special_tokens=["<UNK>", "<PAD>", "<SOS>", "<EOS>"]
		)

		def sentence_iterator():
			for item in dataset:
				yield item["translation"]["en"]
				yield item["translation"]["es"]

		tokenizer.train_from_iterator(sentence_iterator(), trainer=trainer)
		self.tokenizer = tokenizer
		self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)


	def save(self, path: str):
		self.tokenizer.save(path)

	@staticmethod
	def load(path: str) -> 'BPETokenizer':
		t = BPETokenizer()
		t.tokenizer = Tokenizer.from_file(path)
		t.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
		return t

	@property
	def pad_idx(self) -> int:
		return self.tokenizer.token_to_id("<PAD>")

	@property
	def sos_idx(self) -> int:
		return self.tokenizer.token_to_id("<SOS>")

	@property
	def eos_idx(self) -> int:
		return self.tokenizer.token_to_id("<EOS>")

	@property
	def unk_idx(self) -> int:
		return self.tokenizer.token_to_id("<UNK>")

	@property
	def vocab_size_actual(self) -> int:
		return self.tokenizer.get_vocab_size()

	def tokenize(self, text: str) -> list[str]:
		return self.tokenizer.encode(text).tokens

	def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
		ids = self.tokenizer.encode(text).ids
		if add_special_tokens:
			ids = [self.sos_idx] + ids + [self.eos_idx]
		return torch.tensor(ids)

	def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
		if skip_special_tokens:
			special = {self.pad_idx, self.sos_idx, self.eos_idx, self.unk_idx}
			ids = [i for i in ids if i not in special]
		return self.tokenizer.decode(ids)

	def decode_batch(self, indices: torch.Tensor) -> list[str]:
		return self.tokenizer.decode_batch(indices.tolist())

	def __len__(self) -> int:
		return self.vocab_size_actual