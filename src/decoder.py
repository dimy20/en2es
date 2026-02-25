import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxOut(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, X):
		B, _ = X.shape
		return X.view(B, -1, 2).max(dim=-1).values

class Decoder(nn.Module):
	def __init__(self, hidden_size, emb_dim, vocab_size):
		super().__init__()

		self.hidden_size = hidden_size
		self.vocab_size = vocab_size

		self.embeddings = nn.Embedding(vocab_size, emb_dim)
		self.V = nn.Linear(hidden_size, hidden_size, bias=False)

		#update gate
		self.Cz = nn.Linear(hidden_size, hidden_size, bias=False)
		self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
		self.Wz = nn.Linear(emb_dim, hidden_size, bias=False)

		#reset gate
		self.Cr = nn.Linear(hidden_size, hidden_size, bias=False)
		self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
		self.Wr = nn.Linear(emb_dim, hidden_size, bias=False)

		#update params
		self.U = nn.Linear(hidden_size, hidden_size, bias=False)
		self.C = nn.Linear(hidden_size, hidden_size, bias=False)
		self.W = nn.Linear(emb_dim, hidden_size, bias=False)

		# output
		# Im using 2*vocab_size as the target size here since we are using maxout for output
		# keeping it close to what the paper does oringinally.
		# Might change it to something simpler.
		d = 64 # move this to config

		self.Oh = nn.Linear(hidden_size, 2 * d, bias=False)
		self.Oy = nn.Linear(emb_dim, 2 * d, bias=False)
		self.Oc = nn.Linear(hidden_size, 2 * d, bias=False)
		self.max_out = MaxOut()

		self.G = nn.Linear(d, vocab_size, bias=False)
		

	def forward(self, c: torch.Tensor, Y: torch.tensor):
		device = c.device
		h = F.tanh(self.V(c))

		Yemb = self.embeddings(Y)
		B, T, _ = Yemb.shape

		y_logits = torch.zeros(B, T-1, self.vocab_size, device=device)

		for t in range(1, T):
			#update gate
			z = F.sigmoid(self.Wz(Yemb[:, t-1]) + self.Uz(h) + self.Cz(c))
			
			#reset gate
			r = F.sigmoid(self.Wr(Yemb[:, t-1]) + self.Ur(h) + self.Cr(c))

			#candidate update to h
			candidate = F.tanh((self.W(Yemb[:, t-1])) + r * (self.U(h) + self.C(c)))

			#update
			h = z*h + (1-z)*candidate

			# [1, 2*h] => s = [s1, s2, s3, s4, ..... ] #
			# [[s1, s2],
			# [s3, s4]

			# MAX OUT (Might be a good idea to turn this into a module)
			#s_ = h @ self.Oh + Yemb[t-1] @ self.Oy + c @ self.Oc
			#B, H
			s_ = self.Oh(h) + self.Oy(Yemb[:, t-1]) + self.Oc(c)
			s = s_.view(B, -1, 2).max(dim=-1).values

			logits = self.G(s)
			y_logits[:, t-1] = logits

		return y_logits

	def generate(self, c: torch.Tensor, sos_idx: int, eos_idx: int, max_output_tokens : int = 16):
		device = c.device
		#sos_idx = torch.tensor(es_tokenizer.token_to_idx[es_tokenizer.sos_token]).to(device)
		#eos_idx = es_tokenizer.token_to_idx[es_tokenizer.eos_token]
		sos_idx = torch.tensor(sos_idx)

		## NOTE: Preserve Batching
		## This is very subtle, and could cause wrong calculation down the stream
		## we need to add bactching here or otherwise we likely get broadcasting issues and wrong calculations
		y_prev = self.embeddings(sos_idx).unsqueeze(0) ## [1, emb_dim]
		target_tokens = []
		y_token_idx = -1

		h = F.tanh(self.V(c))
		while y_token_idx != eos_idx and len(target_tokens) < max_output_tokens:
			#update gate
			z = F.sigmoid(self.Wz(y_prev) + self.Uz(h) + self.Cz(c))
			#reset gate
			r = F.sigmoid(self.Wr(y_prev) + self.Ur(h) + self.Cr(c))
			#candidate update to h
			candidate = F.tanh(self.W(y_prev) + r * (self.U(h) + self.C(c)))
			#update
			h = z*h + (1-z)*candidate

			# output
			s_ = self.Oh(h) + self.Oy(y_prev) + self.Oc(c)
			s = self.max_out(s)

			#s = s_.view(1, -1, 2).max(dim=-1).values

			y_logits = self.G(s)

			#sample
			probs = F.softmax(y_logits, dim=-1)
			y_token_idx = torch.argmax(probs, dim=-1).item()
			target_tokens.append(y_token_idx)

			y_prev = self.embeddings(torch.tensor(y_token_idx).to(device)).unsqueeze(0) # [1, emb_dim] (Check NOTE)

		return torch.tensor(target_tokens)

