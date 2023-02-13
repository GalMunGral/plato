import os, gzip, json, sys, nltk
from collections import Counter, defaultdict
from utils import load_id_map, get_tokens, normalize_tokens

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

required_dirs = ['.cache', 'model']
for dir in required_dirs:
    os.makedirs(dir, exist_ok=True)

doc_ids = load_id_map('manifest.txt')


def build_stats():
    rev_index = defaultdict(list)
    ref_counter = Counter()

    for i, url in enumerate(doc_ids):
        words = normalize_tokens(get_tokens(url))
        doc_counter = Counter(words)
        ref_counter.update(words)
        for word in doc_counter:
            rev_index[word].append(url)

        with open(f'.cache/{doc_ids[url]}.dat', 'w') as f:
            for word, freq in doc_counter.items():
                f.write(f'{word}:{freq}\n')

        sys.stdout.write(f'\r[{i}] term freq: {url}'.ljust(120))

    unk_freq = 0
    with open('.cache/ref.dat', 'w') as f:
        for word, freq in ref_counter.items():
            if freq < 10:
                unk_freq += freq
            else:
                f.write(f'{word}:{freq}\n')
        f.write(f'<UNK>:{unk_freq}\n')

    with gzip.open(f'index.json.gz', 'wt') as f:
        json.dump(rev_index, f)

    sys.stdout.write('\n')


def build_vocab():
    vocab = []
    with open('.cache/ref.dat') as f:
        while line := f.readline().strip():
            word, _ = line.split(':')
            vocab.append(word)

    with open(f'vocab.txt', 'w') as f:
        for word in sorted(vocab):
            f.write(word + '\n')


def build_unigram(
    file: str,
    vocab: defaultdict[str, int],
    ref: list[float],
    smoothing=0.9,
) -> list[float]:
    counter = defaultdict(int)
    with open(file) as f:
        while line := f.readline().strip():
            word, freq = line.split(':')
            freq = int(freq)
            # either not in vocabulary or rare in document
            if word not in vocab or freq < 5:
                counter['<UNK>'] += freq
            else:
                counter[word] = freq

    lm = [0.0] * len(vocab)
    total = sum(counter.values())
    for word in vocab:
        mle = counter[word] / total
        word_id = vocab[word]
        lm[word_id] = (1 - smoothing) * mle + smoothing * ref[word_id]

    return lm


def build_unigrams() -> None:
    vocab = load_id_map('vocab.txt')
    ref = build_unigram('.cache/ref.dat', vocab, [0.0] * len(vocab))

    for i, (url, id) in enumerate(doc_ids.items()):
        unigram = build_unigram(f'.cache/{doc_ids[url]}.dat', vocab, ref)
        with open(f'model/{id}.dat', 'w') as f:
            f.write(''.join([repr(p).ljust(22) for p in unigram]))
        sys.stdout.write(f'\r[{i}] unigram: {url}'.ljust(120))
    sys.stdout.write('\n')


if __name__ == "__main__":
    # build_stats()
    # build_vocab()
    build_unigrams()
