from dataclasses import dataclass
from src.tokenizer import Tokenizer

@dataclass
class Config:
	hidden_size: int
	emb_dim: int # 500 in paper
	encoder_vocab_size: int
	decoder_vocab_size: int
	batch_size: int
	pad_idx: int
	max_out_dim: int

	@staticmethod
	def default_config(en_tokenizer: Tokenizer, es_tokenizer: Tokenizer) -> 'Config':
		cfg = Config(
			hidden_size = 32,
			emb_dim = 64,
			max_out_dim=64,
			batch_size = 32,
			encoder_vocab_size = en_tokenizer.vocab_size,
			decoder_vocab_size = es_tokenizer.vocab_size,
			pad_idx = en_tokenizer.token_to_idx[en_tokenizer.pad_token]
		)
		return cfg

