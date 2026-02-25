import unittest
from src.encoder import Encoder
from src.decoder import Decoder
import torch

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

if __name__ == '__main__':
	unittest.main()