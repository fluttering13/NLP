import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
# By deriving a set from `raw_text`, we deduplicate the array
CONTEXT_SIZE = 2 
vocab = set(raw_text)
vocab_size = len(vocab)
print('vocab_size',vocab_size)
word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
# generate CBOW data
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return (log_probs)
    
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(vocab), embedding_dim=10, context_size=CONTEXT_SIZE*2)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# print(data)
# print(len(data))#58
for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in data:
        print(context, target)
        #tokenization
        context_ids = make_context_vector(context, word_to_ix)
        #use the gup to compute
        context_ids = context_ids.to(device)
        model.zero_grad()
        log_probs = model(context_ids)
        label = torch.tensor([word_to_ix[target]], dtype=torch.long)
        label = label.to(device)
        #print(context_ids,label)
        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)
print(data[0][0])
test_word=make_context_vector(data[0][0], word_to_ix).to(device)
print(model.embeddings(test_word))
print(model.embeddings(test_word).shape)
print(model(test_word))
label=word_to_ix[data[0][1]]
print(label)
#print(np.argmin((model(test_word).cpu().detach().numpy().flatten())))
