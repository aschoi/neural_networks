import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        """
        Position-wise FFN Constructor
        
        Args:
            d_model:        <int>     Model dimension
            d_ff:           <int>     Feed-forward dimension (typically: 4 * d_model)
            dropout:        <float>   Dropout rate
            activation:     <string>  Activation Function ('relu' or 'gelu')
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation.lower() == 'relu':
            self.activation = F.relu
        elif activation.lower() == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._init_weights()


    def forward(self, X):
        """
        Args:
            X:      <tensor>
        Return:
            X:      <tensor>
        """
        X = self.linear1(X)
        X = self.activation(X)
        X = self.dropout(X)
        X = self.linear2(X)

        return X


    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

