# 主要說明
此處實作主要利用gensism的包處理bible英文語料

## word2vec-gensism.py
訓練模型的code

## gensism_load.py
訓練完模型後

註解部分拿掉可以再訓練模型

從word-emmbedding拿到vectors之後再normalized

就可以計算cos similarity

順便做了PCA分析