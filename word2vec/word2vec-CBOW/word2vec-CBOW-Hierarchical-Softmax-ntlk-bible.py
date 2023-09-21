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

class HuffmanNode:
    def __init__(self, is_leaf, value=None, fre=0, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value  # the node's index in huffman tree
        self.fre = fre  # word frequency in corpus
        self.code = []  # huffman code
        self.code_len = 0  # lenght of code
        self.node_path = []  # the path from root node to this node
        self.left = left  # left child
        self.right = right  # right child
class HuffmanTree:
    def __init__(self, fre_dict):
        self.root = None
        freq_dict = sorted(fre_dict.items(), key=lambda x:x[1], reverse=True)
        self.vocab_size = len(freq_dict)
        self.node_dict = {}
        self._build_tree(freq_dict)
    
    def _build_tree(self, freq_dict):
        '''
            freq_dict is in decent order
            node_list: two part: [leaf node :: internal node]
                leaf node is sorting by frequency in decent order; 
        '''
    
        node_list = [HuffmanNode(is_leaf=True, value=w, fre=fre) for w, fre in freq_dict]  # create leaf node
        node_list += [HuffmanNode(is_leaf=False, fre=1e10) for i in range(self.vocab_size)]  # create non-leaf node

        parentNode = [0] * (self.vocab_size * 2)  # only 2 * vocab_size - 2 be used
        binary = [0] * (self.vocab_size * 2)  # recording turning left or turning right
        
        '''
          pos1 points to currently processing leaf node at left side of node_list
          pos2 points to currently processing non-leaf node at right side of node_list
        '''

        pos1 = self.vocab_size - 1
        pos2 = self.vocab_size
        
        '''
            each iteration picks two node from node_list
            the first pick assigns to min1i
            the second pick assigns to min2i 
            
            min2i's frequency is always larger than min1i
        '''
        min1i = 0
        min2i = 0
        '''
            the main process of building huffman tree
        '''
        for a in range(self.vocab_size - 1):
            '''
                first pick assigns to min1i
            '''
            if pos1 >= 0:
                if node_list[pos1].fre < node_list[pos2].fre:
                    min1i = pos1
                    pos1 -= 1
                else:
                    min1i = pos2
                    pos2 += 1
            else:
                min1i = pos2
                pos2 += 1
            
            '''
               second pick assigns to min2i 
            '''
            if pos1 >= 0:
                if node_list[pos1].fre < node_list[pos2].fre:
                    min2i = pos1
                    pos1 -= 1
                else:
                    min2i = pos2
                    pos2 += 1
            else:
                min2i = pos2
                pos2 += 1
            
            ''' fill information of non leaf node '''
            node_list[self.vocab_size + a].fre = node_list[min1i].fre + node_list[min2i].fre
            node_list[self.vocab_size + a].left = node_list[min1i]
            node_list[self.vocab_size + a].right = node_list[min2i]
            
            '''
                the parent node always is non leaf node
                assigen lead child (min2i) and right child (min1i) to parent node
            '''
            parentNode[min1i] = self.vocab_size + a  # max index = 2 * vocab_size - 2
            parentNode[min2i] = self.vocab_size + a
            binary[min2i] = 1
        
        '''generate huffman code of each leaf node '''
        for a in range(self.vocab_size):
            b = a
            i = 0
            code = []
            point = []

            '''

                backtrace path from current node until root node. (bottom up)
                'root node index' in node_list is  2 * vocab_size - 2 
            '''
            while b != self.vocab_size * 2 - 2:
                code.append(binary[b])  
                b = parentNode[b]
                # point recording the path index from leaf node to root, the length of point is less 1 than the length of code
                point.append(b)
            
            '''
                huffman code should be top down, so we reverse it.
            '''
            node_list[a].code_len = len(code)
            node_list[a].code = list(reversed(code))
            

            '''
                1. Recording the path from root to leaf node (top down). 
                
                2.The actual index value should be shifted by self.vocab_size,
                  because we need the index starting from zero to mapping non-leaf node
                
                3. In case of full binary tree, the number of non leaf node always equals to vocab_size - 1.
                  The index of BST root node in node_list is 2 * vocab_size - 2,
                  and we shift vocab_size to get the actual index of root node: vocab_size - 2
            '''
            node_list[a].node_path = list(reversed([p - self.vocab_size for p in point]))
            
            self.node_dict[node_list[a].value] = node_list[a]
            
        self.root = node_list[2 * vocab_size - 2]

class CBOWDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, windows_size=5, sentence_length_threshold=5):
        self.windows_size = windows_size
        self.sentence_length_threshold = sentence_length_threshold
        self.contexts, self.centers = self._generate_pairs(corpus, windows_size)
        
    def _generate_pairs(self, corpus, windows_size):
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
        return np.array(self.contexts[index]), np.array([self.centers[index]])
    
class HierarchicalSoftmaxLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, freq_dict):
        super().__init__()
        ## in w2v c implement, syn1 initial with all zero
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.syn1 = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=embedding_dim,
            padding_idx=vocab_size
            
        )
        torch.nn.init.constant_(self.syn1.weight.data, val=0)
        self.huffman_tree = HuffmanTree(freq_dict)

    def forward(self, neu1, target):
        # neu1: [b_size, embedding_dim]
        # target: [b_size, 1]
        
        # turns:[b_size, max_code_len_in_batch]
        # paths: [b_size, max_code_len_in_batch]
        turns, paths = self._get_turns_and_paths(target)
        paths_emb = self.syn1(paths) # [b_size, max_code_len_in_batch, embedding_dim]

        loss = -F.logsigmoid(
            (turns.unsqueeze(2) * paths_emb * neu1.unsqueeze(1)).sum(2)).sum(1).mean()
        return loss
    
    def _get_turns_and_paths(self, target):
        turns = []  # turn right(1) or turn left(-1) in huffman tree
        paths = []
        max_len = 0
        ''' we have batch of center words ... '''
        for n in target:
            n = n.item()
            node = self.huffman_tree.node_dict[n]
            
            code = target.new_tensor(node.code).int()  # in code, left node is 0; right node is 1
            turn = torch.where(code == 1, code, -torch.ones_like(code)) # 1 -> 1;  0 -> -1
            
            turns.append(turn)
            '''node_path records the index from root to leaf node in huffman tree'''
            paths.append(target.new_tensor(node.node_path))
            
            if node.code_len > max_len:
                max_len = node.code_len
        
        '''Because each word may has different code length, we should pad them to equal length'''
        turns = [F.pad(t, pad=(0, max_len - len(t)), mode='constant', value=0) for t in turns] 
        paths = [F.pad(p, pad=(0, max_len - p.shape[0]), mode='constant', value=model.hs.vocab_size) for p in paths]
        return torch.stack(turns).int(), torch.stack(paths).long()
    
class CBOWHierarchicalSoftmax(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fre_dict):
        super().__init__()
        self.syn0 = nn.Embedding(vocab_size, embedding_dim)
        self.hs = HierarchicalSoftmaxLayer(vocab_size, embedding_dim, fre_dict)

    
    def forward(self, context, target):
        # context: [b_size, 2 * window_size]
        # target: [b_size]
        neu1 = self.syn0(context.long()).mean(dim=1)  # [b_size, embedding_dim]
        loss = self.hs(neu1, target.long())
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

batch_size=100   
embedding_dim = 50
data_set = CBOWDataset(corpus_indexed)
data_loader = DataLoader(data_set, batch_size=batch_size, num_workers=0)
model = CBOWHierarchicalSoftmax(vocab_size, embedding_dim, fre_dist_indexed)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-5)
File_save_path='./word2vec-CBOW/'

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

torch.save(model, File_save_path+'BIBLE_CBOW_hierachical_softmax.pt')

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
np.save(File_save_path+'cos_list_christ_hs.npy', cos_list_christ)
print(cos_list_christ)

cos_list_jesus=cosineSim.get_synonym('jesus')
np.save(File_save_path+'cos_list_jesus_hs.npy', cos_list_jesus)
print(cos_list_jesus)