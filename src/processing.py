import torch

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
