import jieba
import numpy as np
import random
import heapq
import re
from sklearn.feature_extraction.text import CountVectorizer
pattern=r'\d+|\.|%|—|，|,|【|】|《|》|：|\(|\)|（|）|？|\?|:|\/|\d*%|-|_|;|—|\n|,|，|。|是|；|、|!|即|即便|就|那样|海通|的|广发|\xa0|策略|月|年|上周|速递|宏观|•|‘|’|“|”|●|和|日|有|要|我们|亿|增速|利率|经济|亿|在|又|去|了|我|我们|但|而|任然|万|从|下|可|及|都|占|个|已|姜超|订阅|保持|其中|以来|来看|保持|意味着|一般|分别|研究所|bp'#设定正则过滤方案
step=0
tcount=0
K=10#主题数
alpha=50/K#超参数alpha
beta=0.1#超参数beta
num_words=10#每个主题要展示的10个words
textcount=20#文章数目
itertime=50#迭代次数
words=list()
identity=np.identity(K,dtype='int32')#创建单位矩阵进行给topic进行one-hot编码
TopicName=[]#生成topic
Topic={}#topic字典
topicwords={}#每个topic的word
for k in range(K):
    TopicName.append('Topic'+str(k))
for k,name in zip(range(K),TopicName):
    Topic[name]=identity[k]
words_index=[]#每个主题词的分布
def sample(nwsum, ndsum, nw, nd, topic, textcount,k):
    nw[k][topic]-=1#词的主题分布
    nd[textcount][topic] -= 1  # 每篇文档的主题分布
    nwsum[topic]-=1#主题对应词的总数
    ndsum[textcount] -= 1#每篇文档的主题数
    p=((nd[textcount]+alpha)/(ndsum[textcount]+K*alpha))*((nw[k]+beta)/(nwsum+BOW._shape[1]*beta))
    p=p/np.sum(p)#概率归一
    p = np.ndarray.flatten(p)
    new_topic = np.random.multinomial(1, p)#多项式分布实验做一次获得新的topic
    new_topic_index= np.nonzero(new_topic)[0][0]
    nw[k][new_topic_index]+=1#词的主题分布
    nd[textcount][new_topic_index] += 1  # 每篇文档的主题分布
    nwsum[new_topic_index]+=1#主题对应词的总数
    ndsum[textcount] += 1#每篇文档的主题数
    pre[textcount][k][new_topic_index]+=1
with open('姜超宏观证券研究.txt',encoding='utf-8') as f:
    for text in f.readlines():
        step+=1
        L=((x+1)*3+x for x in range(textcount))
        if (step in L):#for all docs
            tcount+=1
            print(tcount)
            text=re.sub(pattern,'',text)
            seg=jieba.cut(text, cut_all=False)
            words.append(' '.join(seg))#for all words
vectorizer = CountVectorizer()
BOW= vectorizer.fit_transform(words)#生成所有语料库的词袋模型
word_distribute=BOW.toarray()
p=np.zeros(K)#用来存概率
nw=np.zeros((BOW._shape[1],K),dtype='int32')#用来存词的主题分布
nwsum=np.zeros(K)#用来存主题下的总次数
nd=np.zeros((textcount,K),dtype='int32')#某文档的主题分布
ndsum=np.zeros(textcount,dtype='int32')#某文档的总词数
feature_name = vectorizer.get_feature_names()#获取词袋模型的词
pre=np.zeros((textcount,BOW._shape[1],K))#预处理
for i in range(textcount):#随机分配主题
    for k in range(BOW._shape[1]):
        while word_distribute[i][k]:
            a = random.sample(Topic.keys(), 1)
            pre[i][k]=pre[i][k]+Topic[a[0]]
            word_distribute[i][k]-=1
for i in range(textcount):
     nd[i]=np.sum(pre[i],axis=0)
ndsum=np.sum(nd,axis=1)#获得六个主题的文档总词数
nw=np.sum(pre[:],axis=0)#词的主题分布
nwsum=np.transpose(np.sum(nw,axis=0))#主题下的总词数
for i in range(itertime):
    print(i)
    for n in range(textcount):
        for k in range(BOW._shape[1]):#迭代一篇文里的所有词
            s=list(pre[n][k])#获取该词的主题分布
            times=np.sum(s)
            for count in range(int(times)):#若该词不存在于该文章，则直接跳出
                while np.sum(s):#对某一个词，要迭代的次数逐渐减少到0才跳出循环
                    for num in range(K):#判断次的主题分布，进行抽样
                        if s[num]:
                            topic=num
                            s[num]-=1
                            pre[n][k][num]-=1
                            sample(nwsum, ndsum, nw, nd, topic, n, k)
                            continue
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
    value=heapq.nlargest(10, words_index[i])#找到某个主题的具有最大共现次数的10个值
    for k in range(num_words):
        index=np.where(words_index[i] == value[k])[0][0]#找到每个值的下标
        topicwords[topic].append(str(feature_name[index]))#通过feature name把该词添加进去
