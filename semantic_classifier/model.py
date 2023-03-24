from torchtext.vocab import Vocab, GloVe


import torch.nn as nn
import torch.nn.functional as F

from semantic_classifier.utils import create_embedding_weights




class ReviewClassifier(nn.Module):
    def __init__(self, embedding_weights, hidden_size):
        super().__init__()

        vocab_size, embedding_dim = embedding_weights.size()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict({'weight': embedding_weights})
        self.embedding.requires_grad = False

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size, 10)

    def forward(self, seq):
        embedded = self.embedding(seq)
        out, (hidden, _) = self.lstm(embedded)
        preds = self.linear(hidden.squeeze(0))

        return F.softmax(preds, dim=1)
    
    