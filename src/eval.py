import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.model import Seq2Seq
#from src.tokenizer import Tokenizer
from src.tokenizer import BPETokenizer
import sacrebleu
from datasets import load_dataset, DatasetDict

def bleu_references(test_split: DatasetDict, tokenizer: BPETokenizer):
	en, es = [], []
	for row in test_split:
		t = row.get("translation")
		en.append(t.get("en"))
		es.append(t.get("es"))

	# every source sentence in english has a list of rerefence translations in spanish:
	# each sentence can have more that one valid translation, this is used to compute the bleu metric
	# for generated translations.

	groups = {} 
	for k, v in zip(en, es):
		# must match how the tokenizer decodes, otherwise we will get key errors.
		k = " ".join(tokenizer.tokenize(k))
		groups[k] = groups.get(k, []) + [v]

	return groups

# Accuracy on one batch on data.
# Accuracy is not the best metric for machine translation, im going to include it anyways
# need to add BLEU score.
def accuracy(pad_idx: int, Y: torch.Tensor, y_log_probs: torch.Tensor):
	# i dont want to include pad indices in the accuracy
	indices = Y[:, 1:] != pad_idx
	yreal = Y[:, 1:][indices]  # real indices
	ypredicted = torch.argmax(y_log_probs[:, :, :], dim=-1)[indices] # predicted indices
	return torch.mean((yreal == ypredicted).float(), dtype=torch.float)

class ModelEvaluator:
	def __init__(self, 
			  	model: Seq2Seq,
				device: str,
				test_loader: DataLoader,
				tokenizer: BPETokenizer,
			  ):
		self.model = model
		self.device = device
		self.test_loader = test_loader
		self.tokenizer = tokenizer
		test_split = load_dataset("Helsinki-NLP/opus-100", "en-es", split="test")
		self.references = bleu_references(test_split, tokenizer)

	def eval_test(self):
		test_loss = 0.0
		test_acc = 0.0
		avg_bleu_score = 0.0
		self.model.eval()
		with torch.no_grad():
			test_loss = 0

			for Xt, Yt in self.test_loader:
				Xt = Xt.to(self.device)
				Yt = Yt.to(self.device)

				# forward pass on test batch
				y_logits_t = self.model.forward(Xt, Yt)
				y_log_probs_t = F.log_softmax(y_logits_t, dim=-1)
				B, T, Y_VOCAB = y_log_probs_t.shape
				Lt = F.nll_loss(y_log_probs_t.view(B*T, Y_VOCAB), 
							Yt[:, 1:].contiguous().view(-1),
							ignore_index=self.tokenizer.pad_idx)

				test_loss += Lt.item()
				test_acc += accuracy(self.tokenizer.pad_idx, Y=Yt, y_log_probs=y_log_probs_t)
				avg_bleu_score += self.bleu_eval_batch(Xt, self.references)

		self.model.train()
		N = len(self.test_loader)
		
		return test_loss / N, test_acc / N, avg_bleu_score / N

	def bleu_eval_batch(self,
						X: torch.Tensor,
						references: dict) -> float:
		assert not self.model.training, "Model must be in eval mode during bleu_eval_batch"
		with torch.no_grad():
			decoded_X = self.tokenizer.decode_batch(X)
			c = self.model.encoder(X)
			y = self.model.decoder.batch_generate(c, 
										 		  sos_idx=self.tokenizer.sos_idx,
												  eos_idx=self.tokenizer.eos_idx)

			refs_list = [references[x] for x in decoded_X]
			candidates = self.tokenizer.decode_batch(y)

			score = sacrebleu.corpus_bleu(candidates, refs_list).score
			return score
