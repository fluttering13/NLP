# 斷詞(tokenization)
在NLP資料前處理，例如說輸入是一段文字

我們會將文字進行事前的切割與分類，目的是要找出詞與詞之間的關係

切割文章至詞的方法就稱為斷詞


## CkipTagger

第一個工具是CkipTagger，中研院開發的工具

https://github.com/ckiplab/ckiptagger

(WS) word segmentation

(POS) part-of-speech tagging

(NER) named entity recognition

```
pip install ckiptagger
pip install tensorflow
pip install gdown
```

## 結巴Jieba

第二個工具是Jieba，主要是作為斷詞功能

主要利用前綴詞進行詞圖掃描，生成所有成詞情況的有向無環圖(DAG)

並使用動態規劃查找最大概率路徑

Hidden Markov Models可以針對未登錄詞進行處理

```
pip install jieba
pip install paddlepaddle-tiny==1.6.1
```

# N-gram
N-gram的實作放在這裡

# wor2vec
wor2vec共分成兩塊 CBOW 與 skip-gram

並且為了解決softmax運算量太大的問題，有兩種解決辦法

一個是產生二元分類樹來進行有效的資訊分類的hierarchy softmax

另一個是以sampling負分類來節省運算空間的negative sampling
## CBOW

## skip-gram

## negative sampling

## hierarchy softmax


