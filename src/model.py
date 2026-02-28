import torch.nn as nn
import torch
from src.encoder import Encoder
from src.decoder import Decoder
from src.config import Config

class Seq2Seq(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.encoder = Encoder(config.hidden_size, config.emb_dim, config.vocab_size, pad_idx=config.pad_idx)
		self.decoder = Decoder(config.hidden_size, config.emb_dim, config.vocab_size, max_out_dim=config.max_out_dim)

	def forward(self, X: torch.Tensor, Y: torch.Tensor):
		c = self.encoder(X) 
		return self.decoder(c, Y)

	def generate(self, X: torch.Tensor, debug=False):
		c = self.encoder(X)
		return self.decoder.generate(c) if not debug else self.decoder.generate(torch.zeros_like(c))

	@staticmethod
	def load(fname: str) -> 'Seq2Seq':
		device = "cuda" if torch.cuda.is_available() else "cpu"
		data = torch.load(fname, weights_only=False)
		config = data.get("config")
		state_dict = data.get("model_state_dict")
		model = Seq2Seq(config)
		model.load_state_dict(state_dict)
		return model.to(device)