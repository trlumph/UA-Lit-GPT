with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('Довжина датасету в символах:', len(text))

print('Перші 250 символів датасету:', text[:250])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Create a mapping from characters to numbers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[ch] for ch in x])

print(encode('Привіт ЮА-Літ-ГПТ'))
print(decode(encode('Привіт ЮА-Літ-ГПТ')))

import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:250])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
print(train_data[:block_size + 1])

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f'коли вхідний текст: {decode(context.tolist())}, очікуємо наступний символ: {decode([target.tolist()])}')

torch.manual_seed(2023)
batch_size = 4  # how many sequences to process in parallel
block_size = 8  # maximum context length for predictions


def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch('train')
print('Вхідні дані:')
print(xb.shape)
print(xb)
print('Цільові дані:')
print(yb.shape)
print(yb)

print('---------------------')

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension
        context = xb[b, :t + 1]
        target = yb[b, t]
        print(f'коли вхідний текст: {decode(context.tolist())}, очікуємо наступний символ: {decode([target.tolist()])}')

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(2023)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embeddings(idx)  # (B,T,C)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(100):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f'Поточна втрата: {loss.item():.2f}')