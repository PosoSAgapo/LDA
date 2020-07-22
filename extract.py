# -*- coding: UTF-8 -*-
import re
import os
import pandas as pd
import sys
def index_of_str(seq, sub_seq):
    index=[]
    n1=len(seq)
    n2=len(sub_seq)
    for i in range(n1-n2+1):
        #print('seq==%s' % (seq[i:i + n2]))
        if seq[i:i+n2]==sub_seq:
            #print('seq==%s'%(seq[i:i+n2]))
            index.append(i+1)
    return index
keywords=['资金','营业收入','流动比率','毛利率','流动比率','交易','估算','现金'
        ,'利息','算力','估值','下降','增长','价格','现金','利息','利息收入','速动比率'
        ,'利润','营运开销','净利润','同比增长','环比增长','股价上涨']
entity_matching_result=pd.DataFrame(columns=['keyword','relation','extracted_fact','line'])
m=0
filelist=os.listdir('splited_corpus')
for filename in filelist:
    f=open('splited_corpus/'+filename,'r',encoding='utf-8')
    for line in f.readlines():
        resultset=set()
        if '英伟达' in line:
            originline=line
            for key in keywords:
                if key in line:
                    saved_line=line
                    for rs in list(resultset):
                        saved_line=saved_line.replace(rs,'')
                    keyword_position=index_of_str(saved_line, key)
                    nvidia_position=index_of_str(originline, '英伟达')
                    result=re.search('((\d{1,}|\d{2,}\.\d{1,})%)|((\d{1,}|\d{1,}\.\d{1,})(美元|亿美元))',saved_line.replace('\n',''))
                    if result==None:
                        entity_matching_result.loc[m] = ['英伟达', key, saved_line[keywords+1:],originline[nvidia_position[0]-1:]]
                        resultset.add(saved_line[keywords+1:])
                        m+=1
                    else:
                        entity_matching_result.loc[m] = ['英伟达', key, result[0],originline[nvidia_position[0]-1:]]
                        resultset.add(result[0])
                        m+=1
                    print(resultset)
    f.close()