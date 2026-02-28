import torch
import torch.nn.functional as F

#from src.tokenizer import Tokenizer
from src.tokenizer import BPETokenizer
from src.model import Seq2Seq

# PERFORM A SINGLE FORWARD TEST PASS
def single_forward_pass(model: Seq2Seq,
				 		tokenizer: BPETokenizer,
						X: torch.Tensor,
						Y: torch.Tensor):
						

	pad_idx = tokenizer.pad_idx

	print(f"SRC = {X[:1]}")
	print(f"TARGET = {Y[:1]}")
	print(f"Expected initial LOSS -> {-torch.log(torch.tensor(1/tokenizer.vocab_size)):.4f}")

	with torch.no_grad():
		model.eval()
		y_logits = model.forward(X, Y)
		y_log_probs = F.log_softmax(y_logits, dim=-1)
		y_log_probs.shape # 
		B, T, Y_VOCAB = y_log_probs.shape
		#L = F.nll_loss(y_log_probs.view(B*T, Y_VOCAB), Y.view(B * T)) # SKIP <SOS>

		# Batched version of nll_loss call, this function doesnt like batches so we need to adjust the shapes.
		L = F.nll_loss(y_log_probs.view(B*T, Y_VOCAB), 
					Y[:, 1:].contiguous().view(-1),
					#Y[:, 1:].reshape(B * (Y.size(1) - 1)), 
					ignore_index=pad_idx)

		#L = F.nll_loss(y_log_probs[:, ], Y[0, 1:]) # SKIP <SOS>
		print(f"Initial Loss : {L.item():.4f}" )
		model.train()

# Overfitting one example as sanity check: Loss should go down fast
def overfit_one_batch(model: Seq2Seq, X: torch.Tensor, Y: torch.Tensor, pad_idx: int, epochs=300, log_step=10, verbose: bool = False):
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
	losses = []
	for i in range(epochs):
		optimizer.zero_grad()

		y_logits = model.forward(X, Y)
		y_log_probs = F.log_softmax(y_logits, dim=-1)
		B, T, Y_VOCAB = y_log_probs.shape

		# Batched version of nll_loss call, this function doesnt like batches so we need to adjust the shapes.
		L = F.nll_loss(y_log_probs.view(B*T, Y_VOCAB), 
					   Y[:, 1:].contiguous().view(-1),
				       ignore_index=pad_idx)


		if (i + 1) % log_step == 0:
			losses.append(L.item())
			if verbose:
				print(f"Loss: {L.item():.4f}")

		L.backward()

		#clip gradients
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

		optimizer.step()
	return losses

# we can do a teacher forcing pass vs an autogressive pass on an overffited batch
# to confirm the model is predicting correct sequences.
# model should be overffitted on the batch X,Y.
# overfit_one_batch should be executed before this function to create a meaningful test result.
def teacher_forcing_vs_auto_regressive_test(model: Seq2Seq,
											X: torch.Tensor,
											Y: torch.Tensor,
											tokenizer: BPETokenizer):
	with torch.no_grad():
		model.eval()
		x = X[1:2]
		y = Y[1:2]
		print(x.shape, y.shape)

		print("Source:", tokenizer.decode(x[0].cpu().tolist()))
		print("Target:", tokenizer.decode(y[0].cpu().tolist()))

		#encoded src
		c = model.encoder(x)

		# teacher forcing forward
		logits = model.decoder(c, y)
		print("\nTEACHER FORCING SAMPLE:\n")
		s1 = tokenizer.decode(torch.argmax(logits[0], dim=-1).cpu().tolist())
		print(f"{s1}\n")

		# inference autoregressive foward

		sos_idx = tokenizer.sos_idx
		eos_idx = tokenizer.eos_idx

		gen_tokens = model.decoder.generate(c, sos_idx=sos_idx, eos_idx=eos_idx)

		print("\nAUTOREGRESSIVE SAMPLE:\n")
		print(tokenizer.decode(gen_tokens.cpu().tolist()))


		# Check if first token is correct
		first_logits = logits[0, 0]  # logits for first token
		first_pred = torch.argmax(first_logits).item()
		first_target = y[0, 1].item()  # y[0] is <SOS>, y[1] is first real token
		print(f"\nFirst token - predicted: {first_pred} ({tokenizer.idx_to_token.get(first_pred, '???')}), target: {first_target} ({tokenizer.idx_to_token.get(first_target, '???')})")
		model.train()
