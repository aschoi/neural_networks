import torch
from transformer.decoder import TransformerDecoder, TransformerDecoderLayer
from transformer.encoder import TransformerEncoder
from transformer.embeddings import PositionalEncoding, TokenEmbedding
import random


def create_causal_mask(seq_len):
    """
    Create Causal Mask for Decoder Self-Attention
    
    Args:
        seq_len:        <int>     Length representing key-value tensor length
    Return:
        mask_reshaped:  <tensor>  3D tensor of shape (1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))

    return mask.unsqueeze(0)  


# Test Single Transformer Decoder Layer
def test_decoder_layer():
    """
    Return:
        decoder_layer:      <type>
    """
    print("Testing Transformer Decoder Layer")

    # Parameters
    d_model = 64
    num_heads = 8
    d_ff = 256
    src_seq_len = 8
    tgt_seq_len = 6
    batch_size = 2

    # Create Sample Inputs
    torch.manual_seed(42)
    decoder_input = torch.randn(batch_size, tgt_seq_len, d_model)
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)

    # Create Causual Mask for Decoder Self-Attention
    causal_mask = create_causal_mask(tgt_seq_len)

    # Create Decoder Layer
    decoder_layer = TransformerDecoderLayer(d_model, num_heads, d_ff)

    output, (self_attn_weights, cross_attn_weights) = decoder_layer(
        decoder_input, encoder_output, causal_mask
    )

    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {output.shape}")
    print(f"Self-attention weights shape: {self_attn_weights.shape}")
    print(f"Cross-attention weights shape: {cross_attn_weights.shape}")

    # Verify dimensions
    assert output.shape == decoder_input.shape
    assert self_attn_weights.shape == (batch_size, num_heads, tgt_seq_len, tgt_seq_len)
    assert cross_attn_weights.shape == (batch_size, num_heads, tgt_seq_len, src_seq_len)

    return decoder_layer


# Test Complete Encoder-Decoder Interaction
def test_encoder_decoder_interaction():
    """
    Return:
        decoder_output:     <type>
    """
    print("Testing Encoder-Decoder Interaction")

    # Parameters
    vocab_size = 1000
    num_layers = 2
    d_model = 64
    num_heads = 8
    d_ff = 256
    src_seq_len = 10
    tgt_seq_len = 8
    batch_size = 2

    # Create Sample Sequences
    torch.manual_seed(42)
    src_tokens = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

    # Instantiate Components
    token_embedding_comp = TokenEmbedding(vocab_size, d_model)
    positional_encoding_comp = PositionalEncoding(d_model, dropout=0.1) 
    encoder_comp = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
    decoder_comp = TransformerDecoder(num_layers, d_model, num_heads, d_ff)

    # Process Source Sequence through Encoder
    src_embeddings = positional_encoding_comp(token_embedding_comp(src_tokens))
    encoder_output, _ = encoder_comp(src_embeddings)

    # Process Target Sequence through Decoder
    tgt_embeddings = positional_encoding_comp(token_embedding_comp(tgt_tokens))
    causual_mask = create_causal_mask(tgt_seq_len)
    decoder_output, _ = decoder_comp(tgt_embeddings, encoder_output, causual_mask)

    print(f"Source tokens shape: {src_tokens.shape}")
    print(f"Target tokens shape: {tgt_tokens.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {decoder_output.shape}")

    return decoder_output


def collate_fn(batch):
    """Custom collate function for dynamic padding"""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    tgt_output_batch = [item['tgt_output'] for item in batch]

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    tgt_output_batch = torch.nn.utils.rnn.pad_sequence(tgt_output_batch, batch_first=True, padding_value=0)

    return {
        'src': src_batch,
        'tgt': tgt_batch,
        'tgt_output': tgt_output_batch
    }


def main():
    decoder_layer = test_decoder_layer()
    final_output = test_encoder_decoder_interaction()


if __name__ == "__main__":
    main()
