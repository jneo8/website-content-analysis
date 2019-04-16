"""Main.py."""
import sys
import re
import string
import logging
import coloredlogs
from zhon import hanzi, pinyin
import requests
import jieba
from bs4 import BeautifulSoup
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def get_logger(name=__name__):
    """Init func."""
    # Define logger
    logger = logging.getLogger(name)
    # coloredlogs.install(level="DEBUG")
    coloredlogs.install(
        level="INFO",
        logger=logger,
        fmt=(
            '[%(asctime)s] '
            '%(levelname)-5s '
            '%(name)s %(funcName)s %(lineno)d '
            '%(message)s '
        ),
    )
    logger.info(f"Init logger: {name}")
    return logger

logger = get_logger()

def get_context(url, timeout=3):
    """Get raw html.

    If status_code != 200, return False.
    """
    try:
        if not "http" in url:
            url = f"http://{url}"
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        logger.warning(f"Get {url} error, {e}")
        return False

    if not resp.status_code:
        logger.debug(f"Get {url} {resp.status_code}")
        return False

    return resp

def parser(resp):
    """Get html tag content.

    Now we use meta data & title only.
    """
    data = {}
    soup = BeautifulSoup(resp.content, "html.parser")

    # Title
    data["title"] = ""
    title = soup.title
    if title:
        if title.string:
            data["title"] = title.string

    # Meta
    meta = soup.find_all("meta")
    for tag in meta:
        if "name" in tag.attrs.keys() and tag.attrs["name"].strip().lower() in ["description", "keywords"]:
            name= tag.attrs["name"].lower()
            content = tag.attrs.get("content", "")
            if content is not  "":
                data[name] = content

    # href
    links = soup.find_all("a")
    for idx, link in enumerate(links):
        if link.string:
            data[f"link_{idx}"] = link.string

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
    logger.debug(sys.argv)

    ## Get filepath
    if len(sys.argv) < 2:
        logger.debug("Need file name.")
        sys.exit(-1)
    filepath = sys.argv[1]
    logger.debug(f"File: {filepath}")

    ## Get size
    size = 100
    if len(sys.argv) >= 3:
        size = sys.argv[2]
        size = int(size)
    logger.info(f"Size: {size}")

    data = []
    with open(filepath, "r") as domains:
        idx = 0
        for domain in domains.readlines()[:size]:
            idx += 1
            logger.info(f"Crawler: {idx} {domain}")
            domain = domain.strip()
            logger.debug(f"Domain: {domain}")
            resp = get_context(domain, timeout=1)
            if resp:
                info = parser(resp)
                logger.debug(info)
                article = ""
                for k, v in info.items():
                    if v is None:
                        v = ""
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


    ## Interface
    while True:
        idx = input(f"Please input the idx of domain(0 ~ {len(weight) - 1 }): ")
        if idx.isdigit():
            idx = int(idx)
        else:
            print("Please input int")
            continue

        if idx >= len(weight):
            print("Idx out of range.")

        else:
            logger.info(f"Title: {data[idx]['info']['title']}")
            logger.info(f"url: {data[idx]['url']}")

            tmp_weight = []
            for j in range(len(word)):
                if weight[idx][j] > 0:
                    tmp_weight.append(
                            {"word": [word[j]], "weight": weight[idx][j]}
                    )
            tmp_weight = sorted(tmp_weight, key=lambda k: k['weight'], reverse=True)
            for item in tmp_weight[:20]:
                logger.info(item)
            logger.info("\n----------\n")


if __name__ == "__main__":
    main()
