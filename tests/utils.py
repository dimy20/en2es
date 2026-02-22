import torch
# generate a B sequences of random lengths, with element values in the range [0, vocab_size).
def gen_different_length_sequences(batch_size: int, max_length: int,vocab_size: int) -> list:
	X = [ ]
	for _ in range(batch_size):
		length = torch.randint(5, max_length, (1, ))[0].item()
		x = torch.randint(0, vocab_size, (length, ))
		X.append(x)
	return X