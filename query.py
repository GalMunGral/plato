import sys, gzip, json, struct
from math import log2
from collections import Counter, defaultdict
from utils import load_id_map, nlp

doc_ids = load_id_map('manifest.txt')
word_ids = load_id_map('vocab.txt')

with gzip.open('index.json.gz') as f:
    rev_index = json.load(f)


def rank_doc(
    url: str,
    query: Counter[str],
) -> float:
    cross_entropy = 0.0
    with open(f'model/{doc_ids[url]}.dat', 'rb') as f:
        for word, count in query.items():
            f.seek(word_ids[word] * 8)
            (p,) = struct.unpack('d', f.read(8))
            cross_entropy += count * log2(1 / p)
    return cross_entropy


def build_model(query: str) -> Counter[str]:
    doc = nlp(query)
    lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]
    return Counter([x if x in word_ids else '<UNK>' for x in lemmas])


if __name__ == '__main__':
    while True:
        query = input("\nPlease enter: ").strip()
        query_model = build_model(query)

        # candidates = set()
        # for term in query_model:
        #     if term != '<UNK>':
        #         matched = set(rev_index[term])
        #         if candidates:
        #             candidates &= matched
        #         else:
        #             candidates = matched

        # print(f'results: {len(candidates)}')

        results = []
        # for i, url in enumerate(candidates):
        for i, url in enumerate(doc_ids):
            score = rank_doc(url, query_model)
            results.append((score, url))
            sys.stdout.write(f'\r[{i}] ranking: {url}'.ljust(100))
        sys.stdout.write('\r'.ljust(100) + '\n')

        for score, url in sorted(results)[:20]:
            print(f'{url} ({round(score, 4)})')
