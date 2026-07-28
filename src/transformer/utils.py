import torch
import torch.nn as nn


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        """
        Add & Norm Layer Constructor
        Residual Connection + Layer Normalization

        Args:
            d_model:    <int>    Model Dimension
            dropout:    <float>  Dropout rate
        """
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, X, sublayer_output):
        """
        Args:
            X:                  <tensor>  Original input (residual connection)
            sublayer_output:    <tensor>  Output from sublayer (attention or FFN)
        Returns:
            <tensor>  Normalized output after Residual Connection 
        """
        # Add:  Residual Connection + Apply Dropout
        output = X + self.dropout(sublayer_output)

        # Norm: Apply Layer Normalization
        return self.layer_norm(output)

