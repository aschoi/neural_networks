import torch.nn as nn
from transformer.attention import MultiHeadAttention
from transformer.feed_forward import PositionwiseFeedForward
from transformer.utils import AddNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        """      
        Transformer Encoder Layer Constructor

        Args:
            d_model:        <int>     Model dimension
            num_heads:      <type>    Number of Heads
            dropout:        <float>   Dropout rate
            activation:     <string>  Activation Function ('relu' or 'gelu')
        """
        super(TransformerEncoderLayer, self).__init__()
        self.selfAttn_mha_sublayer = MultiHeadAttention(d_model, num_heads, dropout)  # Multi-Head Self-Attention
        self.posWise_feedForward_sublayer = PositionwiseFeedForward(d_model, d_ff, dropout, activation)  # Position-wise FFN
        self.add_norm1_sublayer = AddNorm(d_model, dropout)  # Add & Norm
        self.add_norm2_sublayer = AddNorm(d_model, dropout)  # Add & Norm


    def forward(self, X, mask=None):
        """
        Args:
            X:      <type> 
            mask    <type>
        
        Return:
            X                   <type>
            attention_weights   <type>
        """
        # Multi-Head Self-Attention + Add & Norm
        attn_output, attention_weights = self.selfAttn_mha_sublayer(X, X, X, mask)
        X = self.add_norm1_sublayer(X, attn_output)

        # Position-wise Feed-Forward + Add & Norm
        ff_output = self.posWise_feedForward_sublayer(X)
        X = self.add_norm2_sublayer(X, ff_output)

        return X, attention_weights


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        """
        <Stack> of Transformer Encoder Layers:      stack<TransformerEncoderLayer>

        Args:
            num_layers      <int>     Number of Transformer Encoder Layers
            d_model:        <int>     Model dimension
            num_head:       <type>    Number of Heads
            d_ff            <type>    
            dropout:        <float>   Dropout rate
            activation:     <string>  Activation Function ('relu' or 'gelu')
        """
        super(TransformerEncoder, self).__init__()
        self.transformerE_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, d_ff, dropout, activation
            ) for layer in range(num_layers)
        ])
        self.num_layers = num_layers


    def forward(self, X, mask=None):
        """
        Forward pass through Encoder <stack>

        Args:
            X:      <type>
            mask    <type>
        Return:
            X:                          <type>
            attention_weights_stack     <list <type>>
        """
        attention_weights_stack = []

        for transformer_encoder_layer in self.transformerE_layers:
            X, attn_weights = transformer_encoder_layer(X, mask)
            attention_weights_stack.append(attn_weights)

        return X, attention_weights_stack