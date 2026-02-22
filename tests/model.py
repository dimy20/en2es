import unittest
from src.encoder import Encoder, PaddingMask
from src.decoder import Decoder
from src.processing import BatchProcess
from tests.utils import gen_different_length_sequences
import torch
import torch.testing as tt

HIDDEN_SIZE = 32
EMB_DIM = 64
VOCAB_SIZE = 1024

class ModelTest(unittest.TestCase):
	def test_output_dims(self):
		vocab_size = 1024

		enc = Encoder(
			hidden_size=HIDDEN_SIZE,
			emb_dim=EMB_DIM,
			vocab_size=VOCAB_SIZE,
			pad_idx = 0,
		)

		dec = Decoder(
			hidden_size=HIDDEN_SIZE,
			emb_dim=EMB_DIM,
			vocab_size=VOCAB_SIZE
		)

		Y = torch.randint(1024, size=(8, 16))
		X = torch.randint(1024, size=(8, 16))

		XB, _ = X.shape
		c = enc(X)
		self.assertEqual(X.shape[0], c.shape[0], msg="Encoder should output one hidden representation per input sequence.")

		logits = dec(c, Y)
		B, _, YV = logits.shape

		self.assertEqual(XB, B, msg=f"Decoder should output one target sequence Y for each source sequence X, but {B} != {XB}")
		self.assertEqual(YV, vocab_size, msg=f"Logits should have the same number of elements as the target sequence vocab_size, but got : {YV}")

	def test_padding_mask(self):
		X = gen_different_length_sequences(max_length=20, batch_size=10, vocab_size=512)
		# ununsed in this test, but required by BatchProcess
		Y = gen_different_length_sequences(max_length=20, batch_size=10, vocab_size=512)
		
		batch_process = BatchProcess(pad_idx_src=0, pad_idx_dst=0)
		
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