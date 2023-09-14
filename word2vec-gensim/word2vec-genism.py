from collections import OrderedDict 
import re
import nltk
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gensim
from gensim.models.phrases import Phrases, Phraser
from gensim.test.utils import common_texts
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import pickle
### cast to gpu
path='./word2vec-gensim/'
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
###filter out small frequency
fre_dist = {k : v for k, v in fre_dist.items() if v > 5}

###define the dictionary of index to word and of word to index
vocab_size = len(fre_dist)
idx_to_word = {idx: word for idx,  word in enumerate(fre_dist.keys())}
word_to_idx = {word: idx for idx, word in idx_to_word.items()}
### Transform the corpus to indexed form
corpus_indexed = [[word_to_idx[word] for word in sent if word in word_to_idx] for sent in corpus]
corpus_indexed = [sent for sent in corpus_indexed if len(sent) > 5]
fre_dist_indexed = {word_to_idx[w]: f for w, f in fre_dist.items()}


# class Losslogger(genism.models.callbacks.CallbackAny2Vec):
#     def __init__(self):
#         self.losses=[]
#     def on_epoch_end(self,model):
#         loss=model.get_lastest_loss()
#         self.losses.append(loss)
#         print('Loss after epoch {}:{}'.format(len(self.losses),loss))

def display_pca_scatterplot(model, words):
    # Take word vectors
    word_vectors = np.array([model[w] for w in words])

    # PCA, take the first 2 principal components
    twodim = PCA().fit_transform(word_vectors)[:,:2]

    # Draw
    # plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
    plt.show()



"""
min_count (float, optional) 
– Ignore all words and bigrams with total collected count lower than this value.
"""
"""
threshold (float, optional) 
– Represent a score threshold for forming the phrases (higher means fewer phrases).
A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold. 
Heavily depends on concrete scoring-function, see the scoring parameter.
"""


bigram = Phrases(corpus, min_count=5, threshold=10)
bigram_phraser = Phraser(bigram)
corpus = bigram_phraser[corpus]
output=open(path+'corpus.pkl','wb')
pickle.dump(list(corpus),output)
# trigram = Phrases(corpus, min_count=5, threshold=3)
# trigram_phraser = Phraser(trigram)
# corpus = trigram_phraser[corpus]
# print(list(corpus)[-2])
'''

alpha
The initial learning rate.

negative (int, optional) 
– If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
If set to 0, no negative sampling is used.

max_vocab_size
Every 10 million word types need about 1GB of RAM. Set to None for no limit

sg
Training algorithm: 1 for skip-gram; otherwise CBOW.

negative
if > 0, negative sampling will be used, 
the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
If set to 0, no negative sampling is used.

ns_exponent
the number of subsampling factor the deafult is 0.75

cbow_mean ({0, 1}, optional) 
– If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
'''
# losslogger=Losslogger()

w2v_model = gensim.models.Word2Vec(
        # min_count=3,
        # window=5,
        # size=100,
        # alpha=0.005,
        # min_alpha=0.0007,
        # hs=1,
        # sg=1,
        # workers=4,
        # batch_words=100,
        # cbow_mean = 1
        sentences=None,
        corpus_file=None, 
        vector_size=100, 
        alpha=0.025, 
        window=5, 
        min_count=5, 
        max_vocab_size=None, 
        sample=0.001, 
        seed=1, 
        workers=4, 
        min_alpha=0.0001,
        sg=1, 
        hs=1,
        negative=0, 
        ns_exponent=0.75, 
        cbow_mean=1, 
        epochs=5, 
        null_word=0, 
        trim_rule=None, 
        sorted_vocab=1, 
        batch_words=100, 
        compute_loss=False, 
        callbacks=(), 
        comment=None, 
        max_final_vocab=None, 
        shrink_windows=True
        # callback=[losslogger]        
    )
w2v_model.build_vocab(corpus) # build huffman tree

w2v_model.train(
        corpus,
        total_examples=w2v_model.corpus_count,
        epochs=50,
        report_delay=1)
similarity_christ=w2v_model.wv.most_similar("christ", topn=10)
print(similarity_christ)

###save the model
w2v_model.save(path+'W2V-skipgram-hs-bible.model')
word_vectors = w2v_model.wv
word_vectors.save(path+'W2V-skipgram-hs-bible-vectors.kv')
word_vectors.save_word2vec_format(path+'W2V-skipgram-hs-bible-vectors.txt', binary=False)



words=['faith','jesus_christ','gospel','grace','christ_jesus','world','knowing','sufferings']
display_pca_scatterplot(w2v_model.wv, words)
