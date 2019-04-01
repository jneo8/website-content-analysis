"""Main.py."""
import sys
import re
import string
from zhon import hanzi, pinyin
import requests
import jieba
from bs4 import BeautifulSoup
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

def get_context(url, timeout=3):
    """Get raw html.

    If status_code != 200, return False.
    """
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        print(f"Get {url} error, {e}")
        return False

    if not resp.status_code:
        print(f"Get {url} {resp.status_code}")
        return False

    return resp

def parser(resp):
    """Get html tag content.

    Now we use meta data & title only.
    """
    data = {}
    soup = BeautifulSoup(resp.content, "html.parser")

    # Title
    title = soup.title.string
    data["title"] = title

    # Meta
    meta = soup.find_all("meta")
    for tag in meta:
        if "name" in tag.attrs.keys() and tag.attrs["name"].strip().lower() in ["description", "keywords"]:
            name= tag.attrs["name"].lower()
            data[name] = tag.attrs["content"]
    return data

def clean_text(text):
    """Clean the input text.

    - Use jieba to cut content.
    - Remove special character.
    - Remove multiple space.
    """
    special_charactor = str(hanzi.punctuation + pinyin.punctuation + string.punctuation)
    text = " ".join(jieba.cut(text))
    text = re.sub(f"[{special_charactor}]", "", text)
    text = re.sub(" +", " ", text)
    return text


def main():
    """Main process."""
    print(sys.argv)
    if len(sys.argv) < 2:
        print("Need file name.")
        sys.exit(-1)
    filepath = sys.argv[1]
    print(f"File: {filepath}")

    data = []
    with open(filepath, "r") as domains:
        for domain in domains.readlines():
            domain = domain.strip()
            print(f"Domain: {domain}")
            resp = get_context(domain)
            if resp:
                info = parser(resp)
                article = ""
                for k, v in info.items():
                    info[k] = clean_text(v)
                print(info)
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
