
from collections import OrderedDict 
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import gensim
from gensim.models.phrases import Phrases, Phraser
from gensim.test.utils import common_texts
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import pickle
path='./word2vec-gensim/'
def display_pca_scatterplot(normalized_vectors, words, indexes):
    # Take word vectors
    word_vectors = normalized_vectors

    # PCA, take the first 2 principal components
    pca=PCA(n_components=2)
    pca.fit(normalized_vectors)
    print(pca.explained_variance_ratio_)
    two_d_coordinates= pca.fit_transform(normalized_vectors)
    
    x_list=[]
    y_list=[]
    [x_list.append(two_d_coordinates[index,0]) for index in indexes]
    [y_list.append(two_d_coordinates[index,1]) for index in indexes]
    lable_x=list(map(lambda x:x*1.05,x_list))
    lable_y=list(map(lambda x:x*1.05,y_list))
    print(words)
    # # Draw
    plt.scatter(x_list, y_list, edgecolors='k', c='r')

    [plt.text(lable_x[i],lable_y[i], words[i]) for i in range(len(words))]
    # # for word, (x,y) in zip(words, twodim):
    # #     plt.text(x+0.05, y+0.05, word)
    plt.show()

############# continue to train
# f=open(path+'corpus.pkl', 'rb')
# corpus=pickle.load(f)
# w2v_model=gensim.models.Word2Vec.load(path+'W2V-skipgram-hs-bible.model')
# w2v_model.train(
#         corpus,
#         total_examples=w2v_model.corpus_count,
#         epochs=50,
#         report_delay=1)
# w2v_model.save(path+'W2V-skipgram-hs-bible.model')
#############
keyed_vectors = KeyedVectors.load(path+'W2V-skipgram-hs-bible-vectors.kv')

nor_vecs=keyed_vectors.get_normed_vectors()
# print(nor_vecs.shape)
test_word='jesus'
most_similarity_jesus=keyed_vectors.most_similar(test_word,topn=20)
print(most_similarity_jesus)
words=[]
words.append(test_word)
[words.append(key) for key,cos in most_similarity_jesus]
indexes=[]
[indexes.append(keyed_vectors.get_index(words[i])) for i in range(len(words))]

# words=['faith','jesus_christ','gospel','grace','christ_jesus','world','knowing','sufferings']
display_pca_scatterplot(nor_vecs, words, indexes)