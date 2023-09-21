import itertools
from collections import OrderedDict 
import re
import nltk
import torch
import numpy as np
# import torch.functional as F
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
### cast to gpu
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
bible_file_name=nltk.corpus.gutenberg.fileids()[3]
# bible_text=gutenberg.raw(bible_file_name)
samples  = nltk.corpus.gutenberg.sents(bible_file_name)

pattern = re.compile("[A-Za-z]+")
stop_w =  set(nltk.corpus.stopwords.words('english'))

###filter
corpus = []
for sent in samples:
    sent = [w.lower() for w in sent]
    sent = [w for w in sent if w not in stop_w]
    sent = [w.replace('\n', ' ') for w in sent]
    sent = [w for w in sent if pattern.fullmatch(w)]
    if len(sent) > 5:
        corpus.append(sent)
###build fre dist
fre_dist = nltk.probability.FreqDist()
for sent in corpus:
    fre_dist.update(sent)
###filter out small frequncy
fre_dist = {k : v for k, v in fre_dist.items() if v > 5}

###define the dictionary of index to word and of word to index
vocab_size = len(fre_dist)
idx_to_word = {idx: word for idx,  word in enumerate(fre_dist.keys())}
word_to_idx = {word: idx for idx, word in idx_to_word.items()}
### Transfrom the corpus to indexed form
corpus_indexed = [[word_to_idx[word] for word in sent if word in word_to_idx] for sent in corpus]
corpus_indexed = [sent for sent in corpus_indexed if len(sent) > 5]
fre_dist_indexed = {word_to_idx[w]: f for w, f in fre_dist.items()}

class CBOWDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, windows_size=2, sentence_length_threshold=5):
        self.windows_size = windows_size
        self.sentence_length_threshold = sentence_length_threshold
        self.contexts, self.centers = self.generate_pairs(corpus, windows_size)
        
    def generate_pairs(self, corpus, windows_size):
        contexts = []
        centers = []
        
        for sent in corpus:
            if len(sent) < self.sentence_length_threshold:
                continue
            
            for center_word_pos in range(len(sent)):
                context = []
                for w in range(-windows_size, windows_size + 1):
                    context_word_pos = center_word_pos + w
                    if(0 <= context_word_pos < len(sent) and context_word_pos != center_word_pos):
                        context.append(sent[context_word_pos])
                if(len(context) == 2 * self.windows_size):
                    contexts.append(context)
                    centers.append(sent[center_word_pos])
        return contexts, centers
    
    def __len__(self):
        return len(self.centers)
    
    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.contexts[index])), torch.from_numpy(np.array([self.centers[index]]))
    
class CBOWSoftmax(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.syn0 = nn.Embedding(vocab_size, embedding_dim)
        self.syn1 = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context, center):
        ### b_size is bacthes size (one can train with multiple sentence)
        ### v_size is total number of vocabulary
        ### windows_size is enconding dimesion
        ###  context: [b_size, windows_size]
        ###  center: [b_size, 1]
        embds = self.syn0(context).mean(dim=1) ### mean over orginal sentence size [b_size, embedding_dim]
        out = self.syn1(embds) ### [b_size, v_size]
        log_probs = F.log_softmax(out, dim=1)
        loss = F.nll_loss(log_probs, center.view(-1), reduction='mean')
        return loss

class CosineSimilarity:
    def __init__(self, word_embedding, idx_to_word_dict, word_to_idx_dict):
        self.word_embedding = word_embedding # normed already
        self.idx_to_word_dict = idx_to_word_dict
        self.word_to_idx_dict = word_to_idx_dict
        
    def get_synonym(self, word, topK=10):
        idx = self.word_to_idx_dict[word]
        embed = self.word_embedding[idx]
        
        cos_similairty = w2v_embedding @ embed
        
        topK_index = np.argsort(-cos_similairty)[:topK]
        pairs = []
        for i in topK_index:
            w = self.idx_to_word_dict[i]
#             pairs[w] = cos_similairty[i]
            pairs.append((w, cos_similairty[i]))
        return pairs
    

### defind the model
EMBEDDING_DIM = 50
File_save_path='./word2vec-CBOW/'
batch_size=100

model = CBOWSoftmax(vocab_size, EMBEDDING_DIM)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-6)
dataset = CBOWDataset(corpus_indexed)
data_loader = list(DataLoader(dataset, batch_size=batch_size, num_workers=0))

# ###training

for epoch_i in range(10):
    total_loss = 0
    model.train()
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (context, center) in enumerate(tk0):
        context=context.long().to(device)
        center=center.long().to(device)
        loss = model(context, center)
        model.zero_grad()
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()
        if i==0:
            tk0.set_postfix(loss = total_loss/batch_size)

torch.save(model, File_save_path+'BIBLE_CBOW_softmax.pt')

###fetch word embedding
###vecors are learned in the weights of model.
###Try cos similairity to find the word connection
syn0 = model.syn0.weight.data.cpu()

w2v_embedding = syn0 
w2v_embedding = w2v_embedding.numpy()

l2norm = np.linalg.norm(w2v_embedding, 2, axis=1, keepdims=True)
w2v_embedding = w2v_embedding / l2norm

cosineSim = CosineSimilarity(w2v_embedding, idx_to_word, word_to_idx)

cos_list_christ=cosineSim.get_synonym('christ')
np.save(File_save_path+'cos_list_christ.npy', cos_list_christ)
print(cos_list_christ)

cos_list_jesus=cosineSim.get_synonym('jesus')
np.save(File_save_path+'cos_list_jesus.npy', cos_list_jesus)
print(cos_list_jesus)