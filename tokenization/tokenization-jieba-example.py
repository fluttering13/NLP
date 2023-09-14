import jieba
import jieba.analyse
from optparse import OptionParser
import sys
###增加符號範圍
###jieba.re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-°]+)", re.U)
###這邊可以設置主詞典
###https://github.com/samejack/sc-dictionary
#jieba.set_dictionary('./userdict.txt')

###讀取增加辭典
jieba.load_userdict('./dict/userdict2.txt')
text='台灣機車普及化，拿到駕照的人也越來越多，因此也漸漸出現了一些「三寶」，而在山道上又有另一種稱呼，車圈的人都稱他們為「車界猴子」或是「山道猴子」，常見行為像是過彎吃對向、掛豬肉、看到追焦手就高潮等等，都是有可能發生的危險事。'

# list_text=list(jieba.cut(text, cut_all=False, HMM=True))
# list_cut_all = list(jieba.cut(text, cut_all=True))
list_precise = list(jieba.cut(text, cut_all=False))
# list_search=list(jieba.cut_for_search(text))

# print(list_text)
# print(list_cut_all)
# print(list_precise)
# print(list_search)

print(list_precise)

##############提取關鍵詞IDF算法###########################
###自訂義IDF
###jieba.analyse.set_idf_path(file_name)
###自定義stop words
###jieba.analyse.set_stop_words(file_name)
tags = jieba.analyse.extract_tags(text, topK=5)
print(tags)

###############textank算法###############################
result_rank=jieba.analyse.textrank(text, topK=5, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
print(result_rank)

#################TF IDF##################################
result_tf=jieba.analyse.extract_tags(text, topK=5, withWeight=False, allowPOS=())

###############tokenize##################################
###返回在起始文章的位置
result = jieba.tokenize(text, mode='search')
print(list(result))