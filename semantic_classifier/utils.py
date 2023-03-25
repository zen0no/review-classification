import torch
import numpy as np
from torchtext.vocab import Vocab, GloVe, build_vocab_from_iterator


def create_embedding_weights(vocab: Vocab, embed_dim=300):
    glove = GloVe(name='6B', dim=embed_dim)

    matrix_len = len(vocab)
    weights = np.zeros((matrix_len, embed_dim))

    for i, word in enumerate(vocab):
        try:
            weights[i] = glove[word]
        except KeyError:
            weights[i] = np.random.normal(scale=0.6, size=(embed_dim))
    return torch.tensor(weights), (matrix_len, embed_dim)


def load_vocab_from_text(path: str) -> Vocab:
    with open(path, 'r', encoding='utf-8') as f:
        vocab = build_vocab_from_iterator(([x.rstrip('\n')] for x in f), specials=['<unk>', '<pad>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab
    

