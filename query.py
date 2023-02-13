import sys, gzip, json
from math import log2
from collections import Counter, defaultdict
from utils import load_id_map, normalize_tokens

doc_ids = load_id_map('manifest.txt')
word_ids = load_id_map('vocab.txt')

with gzip.open('index.json.gz') as f:
    rev_index = json.load(f)


def rank_doc(
    url: str,
    query_model: defaultdict[str, float],
) -> float:
    divergence = 0.0
    with open(f'model/{doc_ids[url]}.dat') as f:
        for w, p_q in query_model.items():
            f.seek(word_ids[w] * 22)
            p_d = float(f.read(22))
            divergence += p_q * log2(p_q / p_d)
    return divergence


def build_model(terms: list[str]) -> defaultdict[str, float]:
    normalized = [
        x if x in word_ids else '<UNK>' for x in normalize_tokens(terms)
    ]
    total = len(normalized)
    model = defaultdict(float)
    for word, freq in Counter(normalized).items():
        model[word] = freq / total
    return model


if __name__ == '__main__':
    while True:
        query = input("\nPlease enter: ").strip()
        query_model = build_model(query.split(' '))

        candidates = set()
        for term in query_model:
            if term != '<UNK>':
                matched = set(rev_index[term])
                if candidates:
                    candidates &= matched
                else:
                    candidates = matched

        print(f'results: {len(candidates)}')

        results = []
        for i, url in enumerate(candidates):
            score = rank_doc(url, query_model)
            results.append((score, url))
            sys.stdout.write(f'\r[{i}] ranking: {url}'.ljust(100))
        sys.stdout.write('\r'.ljust(100) + '\n')

        for score, url in sorted(results)[:20]:
            print(f'{url} ({round(score, 4)})')
