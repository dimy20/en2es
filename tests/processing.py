import unittest
from src.processing import BatchProcess

from tests.utils import gen_different_length_sequences

class ProcessingTest(unittest.TestCase):
	def test_batch_padding(self):
		batch_proccess = BatchProcess(pad_idx=0)
		X = gen_different_length_sequences(max_length=20, batch_size=5, vocab_size=512)
		Y = gen_different_length_sequences(max_length=20, batch_size=5, vocab_size=1024)
		padded_X, padded_Y = batch_proccess(list(zip(X, Y)))

		x_lens = [x.shape[0] for x in padded_X]
		max_x_len = max(x_lens)

		self.assertEqual(sum([len(x) == max_x_len for x in padded_X]), len(x_lens))

		y_lens = [y.shape[0] for y in padded_Y]
		max_y_len = max(y_lens)

		self.assertEqual(sum([len(y) == max_y_len for y in padded_Y]), len(y_lens))

if __name__ == '__main__':
	unittest.main()