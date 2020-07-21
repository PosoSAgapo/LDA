import re
import pandas as pd
import sys
sys.path.append('c:\\users\\cbw\\desktop\\ltp-master')
from ltp import LTP
ltp= LTP()
f=open('nvidia2019/sina0_2019.txt','r',encoding='utf-8')
text=f.read()
f.close()
period_split=text.split('。')
segment,hidden=ltp.seg([period_split[4]])
keywords=['资金','营业收入','流动比率','毛利率','净利率','流动比率','交易','估算','现金'
        ,'利息','算力','估值','下降','下降','增长','价格','现金','利息','利息收入','速动比率'
        ,'利润','营运开销']
stop_wordslist=set()
f=open('hit_stopwords.txt','r',encoding='utf-8')
word=f.readlines()
f.close()
for line in range(len(word)):
    if line < 256:
        stopline=word[line].replace('\n','')
        for singe in stopline:
            stop_wordslist.add(singe)
    else:
        stopline = word[line].replace('\n', '')
        stop_wordslist.add(stopline)
f=open('splited_corpus/1.html.txt','r',encoding='utf-8')
for line in f.readlines():
    if '英伟达' in line:
            print(line)
f.close()