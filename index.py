import csv, re, json, sys, nltk
from collections import Counter
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, pos_tag, WordNetLemmatizer
from nltk.lm import Vocabulary, KneserNeyInterpolated, NgramCounter
from nltk.lm.api import LanguageModel
from base64 import b64encode, b64decode
from nltk.corpus import wordnet


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

latin_re = re.compile(r"[a-zA-Z\u00C0-\u017F-]+")


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


def file_id(url: str) -> str:
    return b64encode(url.encode('utf-8')).decode('utf-8')


def read_tokens(file_id) -> list[str]:
    with open(f'cache/{file_id}') as f:
        soup = BeautifulSoup(f, 'html.parser')
        text = soup.get_text()
        return word_tokenize(text)


def build_stats():
    wnl = WordNetLemmatizer()
    col_counter = Counter()

    with open('entries.txt') as f:
        urls = [line.strip() for line in f.readlines()]

    for i, url in enumerate(urls[:5]):
        doc_counter = Counter()
        words = [
            wnl.lemmatize(
                token.lower(),
                wordnet_pos_from_treeback_tag(tag) or wordnet.NOUN,
            )
            for token, tag in pos_tag(read_tokens(file_id(url)))
            if latin_re.fullmatch(token)
        ]
        doc_counter.update(words)
        col_counter.update(words)

        with open(f'data/lemmatized/{file_id(url)}.dat', 'w') as f:
            for word in doc_counter:
                f.write(f'{word}, {doc_counter[word]}\n')

        sys.stdout.write(
            f'\r[{i}] test:{url}, doc: {len(doc_counter)}, col: {len(col_counter)}'.ljust(
                100
            )
        )
    # for i, line in enumerate(lines):
    #     url = line.strip()
    #     id = file_id(url)
    #     with open(f'data/lemmatized/{id}.txt') as f:
    #         while line := f.readline():
    #             vocab.update(line.strip().split(' '))
    #     sys.stdout.write(f'\r[{i}] test:{url}, len = {len(vocab)}'.ljust(100))

    with open(f'data/vocab.dat', 'w') as f:
        for word in col_counter:
            f.write(f'{word}, {col_counter[word]}\n')

    sys.stdout.write('\n')


def build_unigram(id: str, vocab: Vocabulary) -> LanguageModel:
    lm = KneserNeyInterpolated(1, 0.1, vocabulary=vocab)
    text = []
    with open(f'data/lemmatized/{id}.txt') as f:
        while line := f.readline():
            text.append(line.strip().split(' '))
    lm.fit(text)
    return lm


def build_unigrams(vocab: Vocabulary) -> None:
    with open('entries') as list:
        lines = list.readlines()[:1]
        for i, line in enumerate(lines):
            url = line.strip()
            id = file_id(url)
            lm = build_unigram(id, vocab)
            print(lm)


# def build_unigram(text: str) -> None:
#     text_tokens = tokenize.word_tokenize(text)
#     lm = KneserNeyInterpolated(2);
#     lm.fit(
#     unigram = ngrams(text_tokens, 1)

build_stats()


# with open('data/vocab.txt', 'w') as f:
#     for word in sorted(build_vocab()):
#         f.write(word + '\n')
