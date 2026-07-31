import torch
import torch.nn as nn
from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder
from transformer.embeddings import TokenEmbedding, PositionalEncoding


class Transformer(nn.Module):
    def __init__(
            self, 
            src_vocab_size, 
            tgt_vocab_size, 
            src_pad_id=0,
            tgt_pad_id=0,
            d_model=512, 
            num_heads=8, 
            num_encoder_layers=6, 
            num_decoder_layers=6,
            d_ff=2048, 
            max_seq_len=5000, 
            dropout=0.1, 
            activation='relu'
    ):
        """
        Complete Transformer Model for Sequence-To-Sequence Tasks Constructor

        Args:
            src_vocab_size:         <int> 
            tgt_vocab_size:         <int>
            src_pad_id=0:           <int>
            tgt_pad_id=0:           <int>
            d_model:                <int>
            num_heads:              <int>
            num_encoder_layers:     <int> 
            num_decoder_layers:     <int>
            d_ff:                   <int>
            max_seq_len:            <int>
            dropout:                <float>
            activation:             <string>
        """
        super(Transformer, self).__init__()
        self.d_model = d_model

        # Embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model, padding_idx=src_pad_id,)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id,)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder & Decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout, activation)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout, activation)

        # Output Projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()


    def _init_parameters(self):
        """Initialize Model Parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


    def create_padding_mask(self, seq: torch.Tensor, pad_token=0) -> torch.Tensor:
        """
        Create padding mask
        
        Args:
            seq:        <tensor>
            pad_token:  <int>
        Return:
                        <tensor>
        """
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)


    def create_causal_mask(self, size) -> torch.Tensor:
        """
        Create causal mask for decoder Self-Attention

        Args:
            size:       <int>
        Return:
            Boolean mask of shape [1, 1, size, size]
            True means attention is allowed.
        """
        mask = torch.tril(torch.ones(size, size))

        return mask.unsqueeze(0).unsqueeze(1).bool()


    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: torch.Tensor=None, 
        tgt_mask: torch.Tensor=None
    ) -> torch.Tensor:
        """
        Forward Pass Through Complete Transformer
        
        Args:
            src:        <tensor>
            tgt:        <tensor>
            src_mask:   <tensor>
            tgt_mask:   <tensor>
        Return:
            output      <tensor>
        """
        # Embed and encode source
        src_embedded = self.positional_encoding(self.src_embedding(src))  # returns tensor (smallish step)
        encoder_output, _ = self.encoder(src_embedded, src_mask)  # returns tensor (Big Time Step)

        # TransformerEncoder --> TransformerEncoderLayer --> MultiHeadAttention --> AddNorm 
        # Embed and decode target
        # During TRAINING: This is where teacher forcing occurs
        # During INFERENCE: The model gets repetitively fed the sequential generation of its own predicted tokens
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt)) # (smallish step)
        decoder_output, _ = self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)  # (Big time step)

        # Project to Vocabulary
        output = self.output_projection(decoder_output)

        return output
