# TODO: add main function, test model working, integrate with models.py, create visualisations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from sklearn.metrics import roc_auc_score

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Setting up Positional Encoding formula: 
        # PE(pos, i) = sin(pos / (10000)^(i/d_model)) iff i is even
        #            = cos(pos / (10000)^(i/d_model)) iff i is odd

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000) / d_model))      # 1 / (10000)^(i/d_model)

        pe[:, 0::2] = np.sin(position * div_term)       # even form
        pe[:, 1::2] = np.cos(position * div_term)       # odd form

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CRISPRTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model = 128, nhead = 8, num_layers = 2, dropout = 0.1):
        super(CRISPRTransformer, self).__init__()

        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = nn.PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim = 1)         # mean pooling
        x = self.fc(x)
        return x
    
    class SequenceDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels

        # Overloaded dunder methods

        def __len__(self):
            return len(self.sequences)      # or self.labels

        def __getitem__(self, idx):
            return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def train_transformer(model, train_loader, val_loader, num_epochs = 10, lr = 0.1):
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = lr)
    best_auc_score = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimiser.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimiser.step()
            # print(f"Epoch:{epoch}, Loss: {loss.item()}")
            train_loss += loss.item()

    model.eval()
    val_preds, val_probs, val_labels = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim = 1)[:, 1]
            predictions = torch.argmax(outputs, dim = 1)

            # .cpu() calls below ensure that torch tensors (which are usually stored on the GPU are stored in CPU memory before we convert to np arrays (which require CPU memory))
            val_probs.extend(probabilities.cpu().numpy())
            val_preds.extend(predictions.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    val_auc = roc_auc_score(val_labels, val_probs)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}")
            





        