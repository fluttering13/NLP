import itertools
from collections import OrderedDict 
import re
import nltk
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import random, math
from collections import OrderedDict, Counter
import re

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


class NegativeSampler:
    def __init__(self, corpus, sample_ratio=0.75):
        self.sample_ratio = sample_ratio
        self.sample_table =  self.__build_sample_table(corpus)
        self.table_size = len(self.sample_table)
        
    def __build_sample_table(self, corpus):
        counter = dict(Counter(list(itertools.chain.from_iterable(corpus))))
        words = np.array(list(counter.keys()))
        probs = np.power(np.array(list(counter.values())), self.sample_ratio)
        normalizing_factor = probs.sum()
        probs = np.divide(probs, normalizing_factor)
        
        sample_table = []

        table_size = 1e8
        word_share_list = np.round(probs * table_size)
        '''
         the higher prob, the more shares in  sample_table
        '''
        for w_idx, w_fre in enumerate(word_share_list):
            sample_table += [words[w_idx]] * int(w_fre)

#         sample_table = np.array(sample_table) // too slow
        return sample_table
    
    def generate(self, sample_size=6):

        negatvie_samples = [self.sample_table[idx] for idx in np.random.randint(0, self.table_size, sample_size)]
        return np.array(negatvie_samples)
    
class SkipGramWithNGEDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, window_size=5, sentence_length_threshold=5, negative_sample_size=10):
        self.window_size = window_size
        self.sentence_length_threshold = sentence_length_threshold
        self.negative_sample_size = negative_sample_size
        
        self.corpus = self.__subsampling_frequenct_words(corpus)
        self.pairs = self.__generate_pairs(self.corpus, window_size)
        self.negative_sampler = NegativeSampler(self.corpus)
        
    def __sub_sample(self, x, alpha=3):
        pow_ = math.pow(10, alpha)
        s = math.sqrt(x * pow_)
        return (s + 1) / (x * pow_)
    
    def __subsampling_frequenct_words(self, corpus):
        counter = dict(Counter(list(itertools.chain.from_iterable(corpus))))
        sum_word_count = sum(list(counter.values()))
        
        word_ratio ={w: count / sum_word_count  for w, count in counter.items()}
        
        word_subsample_frequency = {k: self.__sub_sample(v) for k, v in word_ratio.items()}
        
        filtered_corpus = [] 
        for sent in corpus:
            filtered_sent = []
            for w in sent:
                if random.random() < word_subsample_frequency[w]:
                      filtered_sent.append(w)
            filtered_corpus.append(filtered_sent)
        return filtered_corpus
    
    
        
    def __generate_pairs(self, corpus, windows_size):     
        pairs = []
        for sentence in corpus:
            if len(sentence) < self.sentence_length_threshold:
                continue

            for center_word_pos in range(len(sentence)):
                for shift in range(-windows_size, windows_size + 1):
                    context_word_pos = center_word_pos + shift
                    
                    if (0 <= context_word_pos < len(sentence)) and context_word_pos != center_word_pos:
                        pairs.append((sentence[center_word_pos], sentence[context_word_pos]))
        return pairs  # [(centerword, a_context_word)]

    def __len__(self):
        return len(self.pairs)
    
    '''
        @return:  1 center_w, 1 context_w, n negative sample words
    '''
    def __getitem__(self, index):
        center_w, context_w = self.pairs[index]

        return np.array([center_w]), np.array([context_w]), self.negative_sampler.generate(self.negative_sample_size)
    
class SkipGramNEG(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.syn0 = nn.Embedding(vocab_size, embedding_dim) # |V| x |K|
        self.neg_syn1 = nn.Embedding(vocab_size, embedding_dim) # |V| x |K|
        torch.nn.init.constant_(self.neg_syn1.weight.data, val=0)
        
    def forward(self, center: torch.Tensor, context: torch.Tensor, negative_samples: torch.Tensor):
        # center : [b_size, 1]
        # context: [b_size, 1]
        # negative_sample: [b_size, negative_sample_num]
        embd_center = self.syn0(center)  # [b_size, 1, embedding_dim]
        embd_context = self.neg_syn1(context) # [b_size, 1, embedding_dim]
        embd_negative_sample = self.neg_syn1(negative_samples) # [b_size, negative_sample_num, embedding_dim]
        
        prod_p =  (embd_center * embd_context).sum(dim=1).squeeze()  # [b_size]
        loss_p =  F.logsigmoid(prod_p).mean() # 1
        
        
        prod_n = (embd_center * embd_negative_sample).sum(dim=2) # [b_size, negative_sample_num]
        loss_n = F.logsigmoid(-prod_n).sum(dim=1).mean() # 1
        return -(loss_p + loss_n)
                                                                                           
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
                                                                                               
EMBEDDING_DIM = 100
batch_size=500
File_save_path='./word2vec-skipgram/'
model = SkipGramNEG(vocab_size, EMBEDDING_DIM)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-6)
dataset = SkipGramWithNGEDataset(corpus_indexed)
data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

for epoch_i in range(4):
    total_loss = 0
    model.train()
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (center, context, neg_samples) in enumerate(tk0):
        context=context.long().to(device)
        center=center.long().to(device)
        neg_samples=neg_samples.long().to(device)       
        loss = model(center, context, neg_samples)
        model.zero_grad()
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()
        if i==0:
            tk0.set_postfix(loss = total_loss)        

torch.save(model, File_save_path+'BIBLE_skipgram_negative_sampling.pt')

###fetch word embedding
###vecors are learned in the weights of model.
###Try cos similairity to find the word connection
syn0 = model.syn0.weight.data.cpu().numpy()
syn1 = model.syn1.weight.data.cpu().numpy()

w2v_embedding = (syn0+syn1)/2
l2norm = np.linalg.norm(w2v_embedding, 2, axis=1, keepdims=True)
w2v_embedding = w2v_embedding / l2norm

cosineSim = CosineSimilarity(w2v_embedding, idx_to_word, word_to_idx)

cos_list_christ=cosineSim.get_synonym('christ')
np.save(File_save_path+'cos_list_christ_ns.npy', cos_list_christ)
print(cos_list_christ)

cos_list_jesus=cosineSim.get_synonym('jesus')
np.save(File_save_path+'cos_list_jesus_ns.npy', cos_list_jesus)
print(cos_list_jesus)