import unittest
import torch
import torch.testing as tt
from src.encoder import PaddingMask
from src.decoder import MaxOut
from src.processing import BatchProcess
from tests.utils import gen_different_length_sequences

HIDDEN_SIZE = 32

class TestMaxout(unittest.TestCase):
	def test_max_out_shape(self):
		B, D = 4, 64
		x = torch.randn(B, D)
		maxout = MaxOut()
		out = maxout(x)
		self.assertEqual(out.shape, (B, D // 2))

	def test_maxout_values(self):
		B = 4
		X = torch.arange(128).view(B, -1)
		expected = X[:, [i for i in range(1, X.shape[1], 2)]]
		maxout = MaxOut()
		maxed_out_X = maxout(X)
		tt.assert_close(maxed_out_X, expected)

	def test_maxed_out_single_batch(self):
		X = torch.arange(4).unsqueeze(0)
		maxout = MaxOut()
		maxed_out_X = maxout(X)
		tt.assert_close(maxed_out_X, torch.tensor([[1, 3]]))

class TestMaskPadding(unittest.TestCase):
	def test_padding_mask(self):
		X = gen_different_length_sequences(max_length=20, batch_size=10, vocab_size=512)
		# ununsed in this test, but required by BatchProcess
		Y = gen_different_length_sequences(max_length=20, batch_size=10, vocab_size=512)
		
		batch_process = BatchProcess(pad_idx=0)
		
		padded_X, _ = batch_process(list(zip(X, Y)))
		B, T = padded_X.shape
		# we need to test that the mask correctly blocks hidden units computed from 
		# x[t] where x[t] = pad_idx.
		# (Actually, the hidden units we want to block are computed from embeddings(pad_idx), but we dont need that to thest here.)

		# The final comptued h at a given timestep t.
		current_h = torch.ones(B, HIDDEN_SIZE)

		# The previously comptued h at t-1.
		prev_h = torch.ones(B, HIDDEN_SIZE) * 0.5

		pad_mask = PaddingMask(
			pad_idx = 0
		)
		x_lengths = torch.tensor([x.shape[0] for x in X]).view(B, 1)
		for i in range(B):
			for t in range(T):
				is_active = padded_X[i, t] != 0
				masked_h = pad_mask(x_lengths, t, current_h, prev_h)
				if is_active:
					tt.assert_close(masked_h[i], current_h[i], msg=f"{current_h[i]} - {masked_h[i]}")
				else:
					tt.assert_close(masked_h[i], prev_h[i], msg=f"{current_h[i]} - {masked_h[i]}")

if __name__ == '__main__':
	unittest.main()