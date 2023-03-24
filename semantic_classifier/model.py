import torch.nn as nn
import torch.nn.functional as F




class ReviewClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.predictor = nn.Linear(hidden_size, 10)

    def forward(self, seq):
        embedded = self.embedding(seq)
        out, (hidden, _) = self.encoder(embedded)
        preds = self.predictor(hidden.squeeze(0))

        return F.softmax(preds, dim=1)
    
    