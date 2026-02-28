from dataclasses import dataclass
from src.tokenizer import Tokenizer

@dataclass
class Config:
	hidden_size: int
	emb_dim: int # 500 in paper
	vocab_size: int
	batch_size: int
	pad_idx: int
	max_out_dim: int

	@staticmethod
	def default_config(vocab_size: int, pad_idx: int) -> 'Config':
		cfg = Config(
			hidden_size = 32,
			emb_dim = 64,
			max_out_dim = 64,
			batch_size = 32,
			vocab_size = vocab_size,
			pad_idx = pad_idx
		)
		return cfg

