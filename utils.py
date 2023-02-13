import re
from collections import defaultdict
from urllib import request
from bs4 import BeautifulSoup
from nltk import (
    word_tokenize,
    pos_tag,
    WordNetLemmatizer,
)
from nltk.corpus import wordnet


latin_re = re.compile(r"[a-zA-Z\u00C0-\u017F][a-zA-Z\u00C0-\u017F-]*")
wnl = WordNetLemmatizer()


def normalize_tokens(tokens: list[str]) -> list[str]:
    return [
        wnl.lemmatize(
            token.lower(),
            wordnet_pos_from_treeback_tag(tag) or wordnet.NOUN,
        )
        for token, tag in pos_tag(tokens)
        if latin_re.fullmatch(token)
    ]


def get_tokens(url: str) -> list[str]:
    soup = BeautifulSoup(request.urlopen(url), 'html.parser')
    text = soup.get_text()
    return word_tokenize(text)


def load_id_map(filename: str) -> defaultdict[str, int]:
    id_map = defaultdict(int)
    with open(filename) as f:
        i = 0
        while item := f.readline().strip():
            id_map[item] = i
            i += 1
    return id_map


def wordnet_pos_from_treeback_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
