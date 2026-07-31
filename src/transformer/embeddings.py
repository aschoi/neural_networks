import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Sinusoidal Positional Encoding

        Args:
            d_model:    <int>    Model Dimension
            max_len:    <int>    Maximum Sequence Length
            dropout:    <float>  Dropout Rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create Positional Encoding Matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Compute the div_term_tensor for Sinusoidal Pattern–
        # div_term shape:  (ceil(d_model / 2), )
        div_term_tensor = torch.exp(
            torch.arange(
                0, d_model, 2, dtype=torch.float32
            ) * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices (0, 2, 4, ...)
        # IF dim_model is even, THEN pe[:, 1::2] has dim_model/2 cols.
        pe[:, 0::2] = torch.sin(position * div_term_tensor)

        # Apply cosine to odd indices (1, 3, 5, ...)
        # IF dim_model odd, THEN pe[:, 1::2] slice has (dim_model-1)/2 cols.
        # div_term has (d_model+1)/2 elements. So div_term[:-1] is used, which has (d_model-1)/2 elements.
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term_tensor[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term_tensor)

        # Add a dimension for "batch" and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, X):
        """
        Add positional encoding to input embeddings
        Args:
            X:      <tensor>
        Return:
            output:     <tensor>
        """
        seq_len = X.size(1)
        X = X + self.pe[:, :seq_len]

        return self.dropout(X)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=0):
        """
        Token Embedding Layer
        Args:
            vocab_size:     <int>   Size of Vocabulary
            d_model:        <int>   Model Dimensions
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx)
        self.d_model = d_model
        nn.init.xavier_uniform_(self.embedding.weight)


    def forward(self, X):
        """
        Convert token indices to embeddings scaled by sqrt(d_model)
        Args:
            X:      <tensor>
        Return:
            output:     <tensor>
        """
        return self.embedding(X) * math.sqrt(self.d_model)
