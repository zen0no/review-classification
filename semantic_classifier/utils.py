import torch
import gdown
import os
import numpy as np
from torchtext.vocab import Vocab, GloVe, build_vocab_from_iterator

import semantic_classifier
from semantic_classifier.model import ReviewClassifier


def create_embedding_weights(vocab: Vocab, embed_dim=300):
    glove = GloVe(name='6B', dim=embed_dim, unk_init=torch.rand_like)

    matrix_len = len(vocab)
    weights = np.zeros((matrix_len, embed_dim))


    for i, word in enumerate(vocab.get_itos()):
        if word == '<pad>':
            weights[i] = np.zeros((embed_dim))
        else:
            weights[i] = glove[word]
    return torch.tensor(weights), (matrix_len, embed_dim)


def load_vocab_from_text(path: str) -> Vocab:
    with open(path, 'r', encoding='utf-8') as f:
        vocab = build_vocab_from_iterator(([x.rstrip('\n')] for x in f), specials=['<unk>', '<pad>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab
    

def load_classifier_model() -> ReviewClassifier:
    url = 'https://drive.google.com/file/d/10h7iwdOki0LrCs6zhKAsWXtFmVjNlUoo/view?usp=sharing'
    out_path = os.path.join(semantic_classifier.__path__[0], 'weights', 'model')
    gdown.download(url, out_path, quiet=False,fuzzy=True)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = load_vocab_from_text(os.path.join(semantic_classifier.__path__[0], 'vocab', 'imdb.vocab'))
    w, _ = create_embedding_weights(vocab=vocab)
    model = ReviewClassifier(w, hidden_size=100)
    model.load_state_dict(torch.load(os.path.join(semantic_classifier.__path__[0], 'weights', 'model'), map_location=torch.device(device)))
    
    return model, vocab

