import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        '''
        Multi-Head Attention Module Constructor

        Args:
            d_model:    <int>    Model Dimension
            num_heads:  <int>    Number of Attention Heads
            dropout:    <float>  Dropout rate
        '''
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear Projections for Query, Key, Value, and output
        self.linearProj_w_Q = nn.Linear(d_model, d_model)
        self.linearProj_w_K = nn.Linear(d_model, d_model)
        self.linearProj_w_V = nn.Linear(d_model, d_model)
        self.linearProj_w_output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self._init_weights_xavier()


    def forward(self, query, key, value, mask=None):
        """
        Args:
            query:      <tensor>
            key:        <tensor>
            value:      <tensor>
            mask:       <tensor>
        Return:
            output:         <tensor>    shape: (batch_size, seq_len_q, dim_model)
            att_weights:    <tensor>    shape: (batch_size, num_heads?, seq_len_q, seq_len_q)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Linear Projections & Reshape for multi-head attention
        Q = self.linearProj_w_Q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linearProj_w_K(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linearProj_w_V(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            if mask.dim() == 2:     # [B, K]
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:   # [B, Q, K] or [1, T, T]
                mask = mask.unsqueeze(1)
            # mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # Apply Attention
        att_output, att_weights = self.scaled_dotProd_attention(Q, K, V, mask)

        # Concatenate heads and apply output projection
        att_output = att_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        output = self.linearProj_w_output(att_output)

        return output, att_weights


    def _init_weights_xavier(self):
        '''
        Initialize weights using Xavier Uniform Initialization 
        '''
        for linear_proj in [self.linearProj_w_Q, self.linearProj_w_K, self.linearProj_w_V, self.linearProj_w_output]:
            nn.init.xavier_uniform_(linear_proj.weight)
            if linear_proj.bias is not None:
                nn.init.constant_(linear_proj.bias, 0)


    def scaled_dotProd_attention(self, q, k, v, mask=None):
        '''
        Scaled Dot Product Attention for Multiple Heads

        Args:
            q:      <tensor>
            k:      <tensor>
            v:      <tensor>
            mask:   <tensor>
        Return:
            output:             <tensor>    shape: (batch_size, num_heads?, seq_len_q, seq_len_q + 1)
            attention_weights:  <tensor>    shape: (batch_size, num_heads?, seq_len_q, seq_len_q)
        '''
        scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.d_k))

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights





