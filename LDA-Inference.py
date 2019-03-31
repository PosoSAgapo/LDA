import jieba
import numpy as np
import random
import heapq
import re
import math
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
def sample(nwsum, ndsum, nw, nd, t, m,word_id,n):
    nd[m][t] -= 1  # 每篇文档的主题分布
    ndsum[m] -= 1#每篇文档的主题数
    p=np.multiply(((nd[m]+alpha)/(ndsum[m]+K*alpha)),((nw[word_id]+beta)/(nwsum+BOW._shape[1]*beta)))
    p=p/np.sum(p)#概率归一
    p = np.ndarray.flatten(p)
    new_topic = np.random.multinomial(1, p)#多项式分布实验做一次获得新的topic
    new_topic_index= np.nonzero(new_topic)[0][0]
    nd[m][new_topic_index] += 1  # 每篇文档的主题分布
    ndsum[m] += 1#每篇文档的主题数
    z[m][n]=new_topic_index
pattern=r'\d+|\.|%|—|，|,|【|】|《|》|：|\(|\)|（|）|？|\?|:|\/|\d*%|-|_|;|—|！|\+|\n|。|是|于|；|、|!|=|．|％|·|"|即|即便|就|那样|海通|的|广发|\xa0|策略|月|年|上周|速递|宏观|•|‘|’|“|”|●|和|日|有|要|我们|亿|增速|利率|经济|亿|在|又|去|了|我|我们|但|而|任然|万|从|下|可|及|都|占|个|已|姜超|订阅|保持|其中|以来|来看|保持|意味着|一般|分别|研究所|bp|所以|因为|如果|本号|平台|观点|意见|进行|研究|任何人|所载|发布|报告|之一|AA|AAA|AAAAA|AAAAAAAA|BP|BPA|BDICCFI|BPAA|BBB|we|which|time|has|consent|except|本号|A'#设定正则过滤方案
with open('parameters.txt','rb') as f:
    nd = pickle.load(f)
    ndsum = pickle.load(f)
    nw = pickle.load(f)
    nwsum = pickle.load(f)
    feature_name = pickle.load(f)
    K = pickle.load(f)
    Topic = pickle.load(f)
    alpha=pickle.load(f)
    beta=pickle.load(f)
itertime=20000
step = 0
tcount = 0
textcount=1
usefulword=[]
words=[]
lenth=[]
with open('test.txt',encoding='utf-8') as f:
    for text in f.readlines():
        step += 1
        L = ((x + 1) * 3 + x for x in range(textcount))
        if (step in L):  # for all docs
            tcount += 1
            print(tcount)
            text = re.sub(pattern, '', text)
            seg = jieba.lcut(text, cut_all=False)
            lenth.append(len(seg))
            for word in feature_name:
                if word in seg:
                    usefulword.append(word)
            words.append(' '.join(usefulword))  # for all words
            usefulword.clear()
vectorizer = CountVectorizer(lowercase=False)
BOW= vectorizer.fit_transform(words)#生成所有语料库的词袋模型
word_distribute=BOW.toarray()
testfeature_name = vectorizer.get_feature_names()#获取词袋模型的词
p=np.zeros(K)#用来存概率
nd=np.zeros((textcount,K),dtype='int32')#某文档的主题分布
ndsum=np.zeros(textcount,dtype='int32')#某文档的总词数
midv=[]
z=[]
for m in range(textcount):
    for k in words[m].split():
        word_id=feature_name.index(k)
        topic_index=random.randint(0,K-1)
        midv.append(topic_index)
        nd[m][topic_index]+=1
        ndsum[m]+=1
    z.append(list(midv))
    midv.clear()

for i in range(itertime):
    print('训练迭代',i)
    for m in range(textcount):
        for n in  range(len(words[m].split())):#迭代一篇文里的所有词
            t=z[m][n]#获取该词的主题分布，同时对应了在文章里是有该词
            word_id=feature_name.index(words[m].split()[n])
            sample(nwsum, ndsum, nw, nd, t, m, word_id,n)

total=np.sum(ndsum)
total1=sum(lenth)
likelihoood=0
word_distribute=BOW.toarray()
for i in range(textcount):
    for m in range(BOW._shape[1]):
        k = feature_name.index(testfeature_name[m])
        likelihoood=likelihoood+(-word_distribute[i][m]*(math.log(np.sum(np.multiply((nw[k]+beta)/(nwsum + BOW._shape[1]*beta),(nd[i]+alpha)/(ndsum[i]+K*alpha))))))
perp=math.exp(likelihoood/total)