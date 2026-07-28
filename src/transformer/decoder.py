import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .utils import AddNorm


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        """
        Transformer Decoder Layer Constructor

        Args:
            d_model:        <int>     Model dimension
            num_heads       <int>     Number of Heads
            d_ff:           <int>    
            dropout:        <float>   Dropout rate
            activation:     <string>  Activation Function ('relu' or 'gelu')

        """
        super(TransformerDecoderLayer, self).__init__()

        # Masked Multi-Head Self-Attention
        self.self_attention_sublayer = MultiHeadAttention(d_model, num_heads, dropout)

        # Encoder-Decoder Multi-Head Attention
        self.cross_attention_sublayer = MultiHeadAttention(d_model, num_heads, dropout)

        # Position-wise FFN
        self.feed_forward_sublayer = PositionwiseFeedForward(d_model, d_ff, dropout, activation)

        # Add & Norm sublayers
        self.add_norm1_sublayer = AddNorm(d_model, dropout)
        self.add_norm2_sublayer = AddNorm(d_model, dropout)
        self.add_norm3_sublayer = AddNorm(d_model, dropout)


    def forward(self, X, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        """
        Args:
            X:                      <tensor>    shape: (batch_size, seq_len_q,   d_model)
            encoder_output:         <tensor>    shape: (batch_size, seq_len_q,   d_model)
            self_attention_mask:    <tensor>    shape: (batch_size,         1, seq_len_q, seq_len_q)
            cross_attention_mask:   <tensor>    shape: (batch_size,         1,         1, seq_len_q)
        Return:
            final_output:           <tensor>    shape: (batch_size, seq_len_q, d_model)
            (self_attn_weights, cross_attn_weights)     tuple(<tensor>, <tensor>)
        """
        # 1) Masked Multi-head Self-Attention + Add & Norm
        self_attn_output, self_attn_weights = self.self_attention_sublayer(X, X, X, self_attention_mask)
        temp_output = self.add_norm1_sublayer(X, self_attn_output)

        # 2) Encoder-Decoder Multi-Head Attention + Add & Norm
        cross_attn_output, cross_attn_weights = self.cross_attention_sublayer(
            temp_output, encoder_output, encoder_output, cross_attention_mask
        )
        temp_output = self.add_norm2_sublayer(temp_output, cross_attn_output)

        # 3) Position-wise FFN + Add & Norm
        ff_output = self.feed_forward_sublayer(temp_output)
        final_output = self.add_norm3_sublayer(temp_output, ff_output)

        return final_output, (self_attn_weights, cross_attn_weights)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        """
        [stack] of Transformer Decoder Layers:    stack<TransformerDecoderLayer>

        Args:
            num_layers      <int>     Number of Transformer Encoder Layers
            d_model:        <int>     Model dimension
            num_heads       <int>    Number of Heads
            d_ff:           <int>    
            dropout:        <float>   Dropout rate
            activation:     <string>  Activation Function ('relu' or 'gelu')
        """
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, num_heads, d_ff, dropout, activation
            ) for layer in range(num_layers)
        ])


    def forward(self, X, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        """
        Args:
            X:                      <tensor>    shape: (batch_size, seq_len_q,   d_model)
            encoder_output:         <tensor>    shape: (batch_size, seq_len_q,   d_model)
            self_attention_mask:    <tensor>    shape: (batch_size,         1, seq_len_q, seq_len_q)
            cross_attention_mask:   <tensor>    shape: (batch_size,         1,         1, seq_len_q)
        Return:
            X:           <tensor>   shape: (batch_size, seq_len_q,   d_model)
            (all_self_attn_weights, all_cross_attn_weights)    tuple(list<tensor>, list<tensor>)    both of shape: (batch_size, num_heads?, seq_len_q, seq_len_q)
        """
        all_self_attn_weights = []
        all_cross_attn_weights = []

        for layer in self.layers:
            X, (self_attn_weights, cross_attn_weights) = layer(
                X, encoder_output, self_attention_mask, cross_attention_mask
            )
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)

        return X, (all_self_attn_weights, all_cross_attn_weights)

        