import re
from collections import defaultdict
from urllib import request
from bs4 import BeautifulSoup
import spacy

nlp = spacy.load(
    'en_core_web_lg',
    enable=[
        'tok2vec',
        'tagger',
        'attribute_ruler',
        'lemmatizer',
    ],
)


def read_tokens(url: str) -> list[str]:
    soup = BeautifulSoup(request.urlopen(url), 'html.parser')
    text = soup.get_text()
    doc = nlp(text)
    return [t.lemma_.lower() for t in doc if t.is_alpha]


def load_id_map(filename: str) -> defaultdict[str, int]:
    id_map = defaultdict(int)
    with open(filename) as f:
        i = 0
        while item := f.readline().strip():
            id_map[item] = i
            i += 1
    return id_map
