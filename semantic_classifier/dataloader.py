from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F
import torch

from spacy.lang.en import English
from torchtext.vocab import Vocab, build_vocab_from_iterator

import os
from typing import Tuple, List

import random


class IMDBReviewDataset(Dataset):
    def __init__(self, root_dir: str, vocab_path: str = None):
        self.root_dir = root_dir

        nlp = English()

        self.tokenizer = nlp.tokenizer

        

        self.classes_label = ['neg', 'pos']

        self._classes_entry_list = [(class_label, entry_name) for class_label in self.classes_label for entry_name in os.listdir(os.path.join(root_dir, class_label))]

        specials = ['<unk>', '<pad>']

        if vocab_path is None:
            self.vocab: Vocab = self._create_vocabulary(specials=specials)
        
        else:
            self.vocab: Vocab = self._import_vocabulary(vocab_path=vocab_path, specials=specials)

        self.vocab.set_default_index(self.vocab['<unk>'])


    def __len__(self):
        return len(self._classes_entry_list)
    

    def _get_examples_token_iterator(self):        
        for entry in self._classes_entry_list:
            with open(os.path.join(self.root_dir, *entry), encoding='utf-8') as f: 
                yield f.read()

    def _create_vocabulary(self, specials):
        return build_vocab_from_iterator(self._get_examples_token_iterator(), specials=specials)

    def _import_vocabulary(self, vocab_path: str, specials):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return build_vocab_from_iterator(([x.rstrip('\n')] for x in f), specials=specials)
        
    

    def __getitems__(self, idx) -> List[Tuple]:
        items = []

        for _id in idx[0]:
                item = self._classes_entry_list[_id]

                with open(os.path.join(self.root_dir, *item), encoding="utf-8") as f:
                    text = f.read()
                    tokens = torch.LongTensor([self.vocab[token.text] for token in self.tokenizer(text)])
                    class_mark = self.classes_label.index(item[0])
                    rating = item[1].split('.')[0].split('_')[-1]

                    items.append((rating, class_mark, tokens))
        return items

    def __getitem__(self, idx) -> Tuple:
        item = self._classes_entry_list[idx]

        with open(os.path.join(self.root_dir, *item), encoding="utf-8") as f:
                text = f.read()
                tokens = torch.LongTensor([self.vocab[token.text] for token in self.tokenizer(text)])
                class_mark = self.classes_label.index(item[0])
                rating = item[1].split('.')[0].split('_')[-1]
                return (rating, class_mark, tokens)
            


class BatchSamplerSimilarLength(Sampler):
    def __init__(self, dataset, batch_size, indicies=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # indicies and lengths
        self.indicies = [(i, len(s[2])) for i, s in enumerate(dataset)]

        if indicies is not None:
            self.indicies = torch.tensor(self.indicies)[indicies].tolist()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indicies)
        
        pooled_indices = []

        for i in range(0, len(self.indicies), self.batch_size * 100):
            pooled_indices.extend(sorted(self.indicies[i:i + self.batch_size * 100], key=lambda x: x[1]))
        self.pooled_indices = [x[0] for x in pooled_indices]

        batches = [self.pooled_indices[i:i + self.batch_size] for i in range(0, len(self.pooled_indices), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

        def __len__(self):
            return len(self.pooled_indices) // self.batch_size
    


def transform_rating(rating):
    encoded = torch.zeros(10)
    encoded[int(rating) - 1] = 1
    return torch.unsqueeze(encoded, 0)


def collate_batch(batch):
    rating_list, label_list, tokens_list = [], [], []

    for (_rating, _label, _tokens) in batch:
        rating_list.append(transform_rating(_rating))
        label_list.append(_label)
        tokens_list.append(_tokens)
    
    return torch.cat(rating_list), torch.unsqueeze(torch.tensor(label_list), 1), pad_sequence(tokens_list, padding_value=0).transpose(0, 1)