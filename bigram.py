import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32  # Size of embeddings for each token
# --------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Unique characters in the text: this is the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters <-> integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    """
    encoder: take a string, output a list of integers
    """
    return [stoi[c] for c in s]


def decode(indices: list[int]) -> str:
    """
    decoder: take a list of integers, output a string
    """
    return "".join([itos[i] for i in indices])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    """
    Generate a batch of data of inputs x and targets y
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack(
        [data[i + 1 : i + block_size + 1] for i in ix]
    )  # Offset by one. These are the targets given the input from the first column to the current one in x.
    return x, y


@torch.no_grad()  # Tell PyTorch that this function will not call backprop. Helps it optimize memory usage
def estimate_loss():
    """
    Average out the loss over multiple batches (`eval_iters`).
    This should be less noisy.
    """
    out = {}
    # Set model into evaluation mode when determining loss
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    # Put model back into training model when exiting
    model.train()

    return out


class BigramLanguageModel(nn.Module):
    """
    Super simple bigram model.
    """

    def __init__(self):
        super().__init__()
        # This model has no behavior difference between training mode and evaluation mode, because it doesn't contain any Batch Norm or Dropout layers.
        # Cleanup: we're now adding a layer of indirection.
        # We have a latent space to transform the tokens into an embedding dimension.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # For every position in the context, we want an encoding. This encoding can be added on to the token embedding.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # A linear layer transforms these embeddings to obtain the logits corresponding to each item in the vocabulary.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # Token embeddings (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # Positional embeddings (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape the logits and targets to be (B*T, C) and (B*T) respectively
            # required by PyTorch cross_entropy loss function
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            # This is crucially different from a different tutorial I did before, where I took the argmax.
            # Taking the argmax would make the model deterministic, and it would always produce the same
            # sequence given the same context.
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):
    # every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
