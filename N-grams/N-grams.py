from collections import Counter, namedtuple
import json
import re
import os
import numpy as np
import pickle

def read_txt(path):
    f = open(path,encoding="utf-8")
    text=f.read()
    return text


def prepocess_text(line: str):
    # 只保留中文字元，並且斷開不連續的中文字
    chinese = r'[\u4E00-\u9FFF]+'
    segments = re.findall(chinese, line)
    return segments

def N_grams_gneration(N,article_processed,Word):
    ngram_prediction = dict()
    total_grams = list()
    words = list()
    # Word = namedtuple('Word', ['word', 'prob'])

    for doc in article_processed:
        split_words =  list(doc)
        ### 統計N個字詞出現的頻率
        [total_grams.append(tuple(split_words[i:i+N])) for i in range(len(split_words)-N+1)]
        ### 統計N-1個字詞出現的頻率
        [words.append(tuple(split_words[i:i+N-1])) for i in range(len(split_words)-N+2)]

    ###建立counter
    total_word_counter = Counter(total_grams)
    word_counter = Counter(words)

    ###把N-gram總表建立一下
    for key in total_word_counter:
        word = ''.join(key[:N-1])
        ###排除掉重複的
        if word not in ngram_prediction:
            ngram_prediction.update({word: set()})
        ###條件機率，在N-1個字出現之下，第N個字出現的機率
        next_word_prob = total_word_counter[key]/word_counter[key[:N-1]]
        
        w = Word(key[-1], '{:.3g}'.format(next_word_prob))
        ngram_prediction[word].add(w)
    return ngram_prediction

N=3
text_name='test-story-1'

path = './N-grams/'+text_name+'_N-grams/'
text_path='./N-grams/'+text_name+'.txt'
save_file_path=path+text_name+'_'+str(N)+'_gram'+'.pkl'

###建立路徑
try:
    os.mkdir(path)
except:
    pass

###主程式：讀取或重新建立檔案
try:
    ##讀取儲存的檔案
    Word = namedtuple('Word', ['word', 'prob'])
    with open(save_file_path, 'rb') as fp:
        ngram_prediction=pickle.load(fp)
    print('read sucessfully')
except:
    ###讀取檔案
    article=read_txt(text_path)
    article_processed=prepocess_text(article)
    ###注意，namedtuple屬性一定要先建立，在functiton裡面會消失
    ngram_prediction=N_grams_gneration(N,article_processed,Word)

    ###按照高低機率重新排列
    for word, ng in ngram_prediction.items():
        ngram_prediction[word] = sorted(ng, key=lambda x: x.prob, reverse=True)

    ###存個檔案
    with open(save_file_path, 'wb') as fp:
        pickle.dump(ngram_prediction, fp)


###把不是1的列出來
# for word, ng in ngram_prediction.items():
#     if ng[0].prob!='1':
#         print(word,ng[0].word,ng[0].prob)

###給定N個字詞，尋出表格
text = '無奈'
next_words = list(ngram_prediction[text])


###N-gram的接龍遊戲
loop_number=100
string_0='無奈'
all_string=string_0
for i in range(loop_number):
    probs_list=[]
    next_word_candiates=[]
    [probs_list.append(float(next_words[i].prob)) for i in range(len(next_words))]
    [next_word_candiates.append(next_words[i].word) for i in range(len(next_words))]
    probs_list=np.array(probs_list)/sum(probs_list)
    next_word=np.random.choice(next_word_candiates,p=probs_list)
    all_string=all_string+next_word
print(all_string)