from torch.utils.data import Dataset


from spacy.lang.en import English

import os
from typing import Tuple



class IMDBReviewDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        nlp = English()

        self.tokenizer = nlp.tokenizer

        

        self.classes_label = ['pos', 'neg']

        self._classes_entry_list = [(class_label, entry_name) for class_label in self.classes_label for entry_name in os.listdir(os.path.join(root_dir, class_label))]

    def __len__(self):
        return len(self._classes_entry_list)
    

    def __getitem__(self, idx) -> Tuple():
        item = self._classes_entry_list[idx]

        with open(os.path.join(self.root_dir, *item)) as f:
            text = f.read()

            tokens = self.tokenizer(text)

            return (tokens, self.tokenizer(text))

    