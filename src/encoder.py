import torch
import torch.nn as nn
import torch.nn.functional as F

# This maks represents:
# 1 -> active sequence (padding token not reached yet at current t)
# 0 -> inactive sequence (reached padding tokens)
# This mask will be used to prevent updates to hidden vector to inactive sequences, preventing the padding index to affect hidden state.
# I call an inactive sequence here at t, if at any given point s[t] = pad_idx.
# s = [4, 5, 6, 7, pad, pad, pad, pad] # up to t = 3 the sequence is active, for t >= 4, inactive.

class PaddingMask(nn.Module):
	def __init__(self, pad_idx):
		super().__init__()
		self.pad_idx = pad_idx

	def forward(self, lengths: torch.Tensor, t: int, h: torch.Tensor, prev_h: torch.Tensor):
		mask = (t < lengths).float().to(h.device)
		block = h * mask
		carry = (1 - mask) * prev_h
		return block + carry

class Encoder(nn.Module):
	def __init__(self, hidden_size, emb_dim, vocab_size, pad_idx):
		super().__init__()
		self.pad_idx = pad_idx
		# used to avoid hidden state calculations from pad_idx tokenes.
		self.padding_mask = PaddingMask(pad_idx)

		self.hidden_size = hidden_size 
		self.vocab_size = vocab_size

		self.embeddings = nn.Embedding(vocab_size, emb_dim)

		self.Wr = nn.Linear(emb_dim, hidden_size)
		self.Ur  = nn.Linear(hidden_size, hidden_size)

		# update gate weights
		self.Wz = nn.Linear(emb_dim, hidden_size)
		self.Uz  = nn.Linear(hidden_size, hidden_size)

		# x-to-h Weights
		self.Wxh = nn.Linear(emb_dim, hidden_size)
		self.Whh = nn.Linear(hidden_size, hidden_size)

		self.V = nn.Linear(hidden_size, hidden_size)

	def forward(self, X: torch.Tensor):
		lengths = torch.sum((X != self.pad_idx).int(), dim=-1, keepdim=True)

		Xemb = self.embeddings(X)
		B, T, _ = Xemb.shape
		h = torch.zeros((B, self.hidden_size), device=X.device)
			
		for t in range(T):
			r = F.sigmoid((self.Wr(Xemb[:, t])) + self.Ur(h)) # the reset gate
			z = F.sigmoid(self.Wz(Xemb[:, t]) + self.Uz(h)) # updated gate
			update = F.tanh(self.Wxh(Xemb[:, t]) + self.Whh(r*h))
			prev_h = h.clone()
			h = z*h + (1-z)*update
			h = self.padding_mask(lengths, t, h, prev_h) # mask inactive sequences

		C = F.tanh(self.V(h))
		return C