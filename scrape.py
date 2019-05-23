# function definition.
import nltk
import string
import re
import numpy as np
import pandas as pd
import pickle
import lda

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction import stop_words

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
from bokeh.transform import factor_cmap

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("lda").setLevel(logging.WARNING)
from bs4 import BeautifulSoup
import re
import requests
import pandas as pd
import pickle
import itertools
import pprint
# from google.colab import files

# collect journal urls from SJR.
html_doc = requests.get('https://www.scimagojr.com/journalrank.php?area=1100&category=1106').text
soup = BeautifulSoup(html_doc, 'html.parser')
tags = soup.select("a")
url_list = []
journal_urls = []

for i in [x.get("href") for x in tags if "journalsearch.php?q=" in x.get("href")]:
    doc = requests.get("https://www.scimagojr.com/" + i).text
    soup = BeautifulSoup(doc, 'html.parser')
    tags = soup.select("a")
    try:
        journal_url = [requests.get(x.get("href")).url for x in tags if x.string=="Homepage"]
        if journal_url==[]:
            journal_url = [""]
        journal_urls = journal_urls + journal_url
        print(counter)
        print(journal_urls)
    except:
        journal_urls = journal_urls + journal_url
        print("connection error.")
        print(counter)
        print(journal_urls)
    counter += 1

f = open('journal_urls.txt', 'wb')
pickle.dump(journal_urls, f)
f = open("./journal_urls.txt","rb")
journal_urls = pickle.load(f)


# make df of journal urls and impact factor scores.
html_doc = requests.get('https://www.scimagojr.com/journalrank.php?area=1100&category=1106').text
soup = BeautifulSoup(html_doc, 'html.parser')
journal_titles = [i.text for i in soup.find_all('a', title='view journal details')]
journal_scores = [i.text.split()[0] for i in soup.find_all('td', class_='orde')]

df = pd.DataFrame({'journal_titles': journal_titles, 'journal_urls': journal_urls, 'journal_scores': journal_scores})

# create 'simple_urls' column to find url from html files.
simple_urls = [x.split("/")[2] if not x == "" and not x.split("/")[2] == "www.elsevier.com" else "https://www.journals.elsevier.com/" + x.split("/")[4] if not x == "" else x for x in journal_urls]
# simple_urls = [x.split("/")[2] if not x == "" or not x.split("/")[2] == www.elsevier.com else x for x in journal_urls]
df['simple_urls'] = simple_urls

# save df
df.to_csv('./journal_urls_if_df.csv')
df1 = pd.read_csv('./journal_urls_if_df.csv')


# collect words of foods in English and Japanese.
html_doc = requests.get('https://www.eigo-love.jp/english-word-list-food/').text
soup = BeautifulSoup(html_doc, 'html.parser')

en_words = [i.string for i in soup.find_all('td', class_='column-1')]
jp_words = [i.string for i in soup.find_all('td', class_='column-2')]

columns = ['englih_words', 'japanese_wprds']
df_words = pd.DataFrame(jp_words,en_words).reset_index()
df_words.columns = columns

html_doc = requests.get('https://eigozuki.com/tango-kenko/byoki.shtml').text
soup = BeautifulSoup(html_doc, 'html.parser')

list_ill = [i.string for i in soup.find_all('td')]
jp_ill = [x for i, x in enumerate(list_ill) if i%2==0]
en_ill = [x for i, x in enumerate(list_ill) if i%2==1]

columns = ['englih_words', 'japanese_wprds']
df_ill = pd.DataFrame(jp_ill,en_ill).reset_index()
df_ill.columns = columns

# create keyword list from food word list and ill list.
en_keyword_list = list(itertools.product(en_words, en_ill))
en_keyword_list = [i[0] + " " + i[1] for i in en_keyword_list]

jp_keyword_list = list(itertools.product(jp_words, jp_ill))
jp_keyword_list = [i[0] + " " + i[1] for i in jp_keyword_list]



jp_aff_list = ["https://px.a8.net/", "https://track.affiliate-b.com/", "ck.jp.ap.valuecommerce.com/", \
               "http://h.accesstrade.net", "https://j-a-net.jp/", "https://hb.afl.rakuten.co.jp", \
              "https://click.linksynergy.com", "https://www.amazon.co.jp", "googleads.g.doubleclick.net", \
              "af.moshimo.com"]

us_aff_list = ["https://www.amazon.com", "revcontent.com", "https://hb.afl.rakuten.com", \
               "www.googleadservices.com", "cat.jp.as.criteo.com"]

ncbi_link = "https://www.ncbi.nlm.nih.gov/"
journal_list = ["https://ci.nii.ac.jp", "elsevier.com"]

def get_search_results_df(keyword):
  columns = ['search_words','rank','title','url', 'affiliate_url', 'paper_url', 'n_affiliate_urls', 'n_paper_urls', 'html']
  df = pd.DataFrame(columns=columns)
  html_doc = requests.get('https://www.google.com/search?num=10&q=' + keyword).text
  soup = BeautifulSoup(html_doc, 'html.parser') # BeautifulSoupの初期化
  tags = soup.find_all('h3',{'class':'r'})
  rank = 1
  for tag in tags:
    title = tag.text
    print (title)
    if "の画像検索結果" in title:
        print("this is a image.")
    else:
        url = query_string_remove(tag.select("a")[0].get("href").replace("/url?q=",""))
        if ".pdf" in url:
            print("this is a pdf file")
        else:
            affiliate_url, n_aff_urls =  get_a8_links(url)
            paper_url, n_paper_urls, html = get_paper_links(url)
            se = pd.Series([keyword, rank, title, url, affiliate_url, paper_url, n_aff_urls, n_paper_urls, html], columns)
            df = df.append(se, ignore_index=True)
            rank += 1
  return df

def query_string_remove(url):
 return url[:url.find('&')]


def get_a8_links(url):
 try:
     html_doc = requests.get(url).text
     soup = BeautifulSoup(html_doc, 'html.parser') # BeautifulSoupの初期化
     tags = soup.select("a")
     urls = ""
     for tag in tags:
       try:
         url = tag.get("href")
         bool = any([True for x in jp_aff_list if url.find(x) > -1]) or any([True for x in us_aff_list if url.find(x) > -1])
         if bool:
           urls += url + "\n"
       except Exception as e:
         continue
     if urls=='':
        n_urls = 0
     else:
         n_urls = len(urls.split('\n'))-1
     return urls, n_urls
 except:
        print("error")
        urls = None
        n_urls = 0
        return urls, n_urls

def get_paper_links(url):
 try:
     html_doc = requests.get(url)
     html_doc.encoding = html_doc.apparent_encoding
     soup = BeautifulSoup(html_doc.text, 'html.parser') # BeautifulSoupの初期化
     tags = soup.select("a")
     urls = ""
     for tag in tags:
        bools = []
        scores = []
        try:
            url = tag.get("href")
            bool = any([False if x=='' else True if url.find(x) > -1 else False for x in df['simple_urls']]) \
                    or url.find(ncbi_link) > -1 or any([True for x in journal_list if url.find(x) > -1])
            if bool:
                urls += url + "\n"
        except Exception as e:
            continue
     if urls=='':
        n_urls = 0
     else:
         n_urls = len(urls.split('\n'))-1
     return urls, n_urls, html_doc.text
 except:
        print("error")
        urls = None
        n_urls = 0
        soup = None
        return urls, n_urls, html_doc


# keywords
columns = ['search_words','rank','title','url', 'affiliate_url', 'paper_url', 'n_affiliate_urls', 'n_paper_urls', 'html']
en_master_df = pd.DataFrame(columns=columns)
keywords = en_keyword_list
for i in keywords:
    print(i)
    search_results_df = get_search_results_df(i)
    en_master_df = pd.concat([en_master_df, search_results_df],ignore_index=True, axis=0)

jp_master_df = pd.DataFrame(columns=columns)
keywords = jp_keyword_list
for i in keywords:
    print(i)
    search_results_df = get_search_results_df(i)
    jp_master_df = pd.concat([jp_master_df, search_results_df],ignore_index=True, axis=0)

# save dataframe as csv.
en_master_df.to_csv('./en_master_df02.csv')
jp_master_df.to_csv('./jp_master_df02.csv')
