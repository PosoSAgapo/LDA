import os
import pickle
import re
import time
import requests
from bs4 import BeautifulSoup
import logging
headers={'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'}#网页请求头
url='http://emweb.eastmoney.com/pc_usf10/FinancialAnalysis/index?color=web&code=NVDA.O'#网页链接
r=requests.get(url,headers=headers)
r.encoding = 'utf-8'#依据网页内容的编码改变编码方式
soup = BeautifulSoup(r.text,features='html5lib')