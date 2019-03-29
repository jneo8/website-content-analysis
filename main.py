"""Main.py."""
import re
import string
from zhon import hanzi, pinyin
import requests
import jieba
from bs4 import BeautifulSoup
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

def get_context(url):
    try:
        resp = requests.get(url, timeout=3)
    except:
        return False

    if not resp.status_code:
        return False

    return resp

def parser(resp):
    data = {}
    soup = BeautifulSoup(resp.content, "html")

    # Title
    title = soup.title.string
    data["title"] = title

    meta = soup.find_all("meta")
    for tag in meta:
        if "name" in tag.attrs.keys() and tag.attrs["name"].strip().lower() in ["description", "keywords"]:
            name= tag.attrs["name"].lower()
            data[name] = tag.attrs["content"]
    return data

def clean_text(text):
    special_charactor = str(hanzi.punctuation + pinyin.punctuation + string.punctuation)
    text = " ".join(jieba.cut(text))
    text = re.sub(f"[{special_charactor}]", "", text)
    text = re.sub(" +", " ", text)
    return text


def main():
    domains = [
        "https://youtube.com",
        "https://www.pixnet.net/",
        "https://www.techbang.com/",
        "https://tw.news.yahoo.com/",
        "https://news.ebc.net.tw/",
        "https://udn.com/news/breaknews/1",
        "https://www.gamer.com.tw/",
        "https://www.104.com.tw/jobs/main/",
        "https://world.taobao.com/",
        "https://www.instagram.com/",
        "https://www.ebay.com/",
        "https://www.microsoft.com/zh-tw/",
        "https://www.paypal.com/us/home",
        "https://github.com/",
        "https://www.whatsapp.com/",
        "https://www.hbo.com/",
    ]
    print(len(domains))

    data = []

    for domain in domains:
        resp = get_context(domain)
        if resp:
            info = parser(resp)
            article = ""
            for k, v in info.items():
                info[k] = clean_text(v)
            data.append(
                {
                    "url": resp.url,
                    "article": " \n ".join(info.values()),
                    "info": info,
                }
            )

    articles = [i["article"] for i in data]

    # TFIDF
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(articles))

    word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    for i in range(len(weight)):
        print(f"Title: {data[i]['info']['title']}")
        print(f"url: {data[i]['url']}")
        print("\n\n")
        tmp_weight = []
        for j in range(len(word)):
            if weight[i][j] > 0:
                tmp_weight.append(
                        {"word": [word[j]], "weight": weight[i][j]}
                )
        tmp_weight = sorted(tmp_weight, key=lambda k: k['weight'], reverse=True)
        for item in tmp_weight:
            print(item)
        print("\n----------\n")


if __name__ == "__main__":
    main()
