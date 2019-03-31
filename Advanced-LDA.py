import jieba
import numpy as np
import random
import heapq
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import random
pattern=r'\d+|\.|%|—|，|,|【|】|《|》|：|\(|\)|（|）|？|\?|:|\/|\d*%|-|_|;|—|！|\+|\n|。|是|于|；|、|!|=|．|％|·|"|即|即便|就|那样|海通|的|广发|\xa0|策略|月|年|上周|速递|宏观|•|‘|’|“|”|●|和|日|有|要|我们|亿|增速|利率|经济|亿|在|又|去|了|我|我们|但|而|任然|万|从|下|可|及|都|占|个|已|姜超|订阅|保持|其中|以来|来看|保持|意味着|一般|分别|研究所|bp|所以|因为|如果|本号|平台|观点|意见|进行|研究|任何人|所载|发布|报告|之一|AA|AAA|AAAAA|AAAAAAAA|BP|BPA|BDICCFI|BPAA|BBB|we|which|time|has|consent|except|本号|A'#设定正则过滤方案
step = 0
tcount = 0
K=30#主题数
alpha=50/K#超参数alpha
beta=0.1#超参数beta
num_words=20#每个主题要展示的10个words
textcount=40#训练文章数目
itertime=55#迭代次数
words=list()
TopicName=[]#生成topic
Topic={}#topic字典
topicwords={}#每个topic的word
words_index=[]#每个主题词的分布
seg1=[]
for k in range(K):
    TopicName.append('Topic'+str(k))
def sample(nwsum, ndsum, nw, nd, t, m,word_id,n):
    nw[word_id][t]-=1#词的主题分布
    nd[m][t] -= 1  # 每篇文档的主题分布
    nwsum[t]-=1#主题对应词的总数
    ndsum[m] -= 1#每篇文档的主题数
    p=np.multiply(((nd[m]+alpha)/(ndsum[m]+K*alpha)),((nw[word_id]+beta)/(nwsum+BOW._shape[1]*beta)))
    p=p/np.sum(p)#概率归一
    p = np.ndarray.flatten(p)
    new_topic = np.random.multinomial(1, p)#多项式分布实验做一次获得新的topic
    new_topic_index= np.nonzero(new_topic)[0][0]
    nw[word_id][new_topic_index]+=1#词的主题分布
    nd[m][new_topic_index] += 1  # 每篇文档的主题分布
    nwsum[new_topic_index]+=1#主题对应词的总数
    ndsum[m] += 1#每篇文档的主题数
    z[m][n]=new_topic_index
with open('姜超宏观证券研究.txt',encoding='utf-8') as f:
    for text in f.readlines():
        step+=1
        L=((x+1)*3+x for x in range(textcount))
        if (step in L):#for all docs
            tcount+=1
            print(tcount)
            text=re.sub(pattern,'',text)
            seg=jieba.cut(text, cut_all=False)
            seg1.append(jieba.lcut(text,cut_all=False))
            words.append(' '.join(seg))#for all words
vectorizer = CountVectorizer(token_pattern='\\b\\w+\\b',lowercase=False)
BOW= vectorizer.fit_transform(words)#生成所有语料库的词袋模型
word_distribute=BOW.toarray()
p=np.zeros(K)#用来存概率
nw=np.zeros((BOW._shape[1],K),dtype='int32')#用来存词的主题分布
nwsum=np.zeros(K)#用来存主题下的总词数
nd=np.zeros((textcount,K),dtype='int32')#某文档的主题分布
ndsum=np.zeros(textcount,dtype='int32')#某文档的总词数
midv=[]
z=[]
feature_name = vectorizer.get_feature_names()#获取词袋模型的词
for m in range(textcount):
    for k in words[m].split():
        word_id=feature_name.index(k)
        topic_index=random.randint(0,K-1)
        midv.append(topic_index)
        nw[word_id][topic_index]+=1
        nwsum[topic_index]+=1
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

multitopicflag=np.zeros((1,textcount),dtype='int32')#一篇文章是否由多主题生成
maintopicflag=np.zeros((1,textcount),dtype='int32')#主要主题是第几个主题
for i in range(textcount):
    flag=-1
    maintopicflag[0][i]=np.argmax(nd[i])
    for k in range(K):
        if (nd[i][k]>100) :#在主要主题外某一个主题的词数超过一百我们则认为这篇文本是多主题生成的
            flag+=1
        if flag>=1:
            multitopicflag[0][i]=1
for k,name in zip(range(K),TopicName):#
    topicwords[name]=list()
words_index=nw.transpose()#将词的主题分布进行转置，得到主题的词分布矩阵
for i,topic in zip(range(K),list(topicwords.keys())):
    value=heapq.nlargest(num_words, words_index[i])#找到某个主题的具有最大共现次数的num_words个值
    valuedict=Counter(value)
    value=sorted(valuedict.keys(),reverse=True)
    for k in value:
        for p in range(valuedict[k]):
            index=np.where(words_index[i] ==k)[0][int(p)]#找到每个值的下标
            topicwords[topic].append(str(feature_name[index]))#通过feature name把该词添加进去
phi=((nw+beta)/(nwsum+BOW._shape[1]*beta)).transpose()
theta=(nd+alpha)/(np.sum(ndsum)+K*alpha)
import pickle
with open('parameters.txt','wb') as f:
    pickle.dump(nd,f)
    pickle.dump(ndsum, f)
    pickle.dump(nw, f)
    pickle.dump(nwsum, f)
    pickle.dump(feature_name, f)
    pickle.dump(K, f)
    pickle.dump(Topic, f)
    pickle.dump(alpha, f)
    pickle.dump(beta, f)
