import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time

from .model import Transformer
from .train import TransformerTrainer
from datasets import load_dataset

from itertools import chain
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from pathlib import Path


# tokenizer
def tokenize(dataset):
    tokenizer = Tokenizer(
        BPE(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=8000,
        min_frequency=2,
        special_tokens=[
            '[PAD]', '[UNK]', '[BOS]', '[EOS]', 
        ], 
    )

    training_texts = chain(dataset['en'], dataset['de'])
    tokenizer.train_from_iterator(
        training_texts, trainer=trainer,
    )

    return tokenizer

# def main():
#     # Test Model Inference

#     def collate_fn(batch):
#         """
#         Custom collate function
#         Args:

#         Return:
#         """
#         src_seqs = []
#         tgt_seqs = []

#         for sample in batch:
#             src_ids = tokenizer.encode(sample['en']).ids
#             tgt_ids = tokenizer.encode(sample['de']).ids
#             src = torch.tensor([BOS_ID, *src_ids, EOS_ID], dtype=torch.int64)
#             tgt = torch.tensor([BOS_ID, *tgt_ids, EOS_ID], dtype=torch.int64)

#             src_seqs.append(src)
#             tgt_seqs.append(tgt)

#         src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=PAD_ID)
#         tgt_batch = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_ID)

#         return {
#             "src": src_batch,
#             "tgt": tgt_batch[:, :-1],
#             "tgt_output": tgt_batch[:, 1:],
#             "src_padding_mask": src_batch.eq(PAD_ID),
#             "tgt_padding_mask": tgt_batch[:, :-1].eq(PAD_ID),
#         }


#     print("Testing Transformer Inference.")

#     dataset = load_dataset("bentrevett/multi30k")

#     print(type(dataset))
#     print(dataset["train"][0])
#     print(dataset['train'])
#     print(dataset)

#     # training_dataset = dataset['train']
#     validation_dataset = dataset['validation']
#     test_dataset = dataset['test']

#     tokenizer = tokenize(test_dataset)

#     PAD_ID = tokenizer.token_to_id("[PAD]")
#     UNK_ID = tokenizer.token_to_id("[UNK]")
#     BOS_ID = tokenizer.token_to_id("[BOS]")
#     EOS_ID = tokenizer.token_to_id("[EOS]")

#     VOCAB_SIZE = tokenizer.get_vocab_size()  

#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=16, 
#         shuffle=True, 
#         collate_fn=collate_fn
#     )
    
#     # Create Model
#     model = Transformer(
#         src_vocab_size=VOCAB_SIZE,
#         tgt_vocab_size=VOCAB_SIZE,
#         d_model=256,
#         num_heads=8,
#         num_encoder_layers=3,
#         num_decoder_layers=3,
#         d_ff=512,
#         dropout=0.1
#     )

#     checkpoint_path = Path("checkpoints/english2German_latest.pt")
#     checkpoint = torch.load(
#         checkpoint_path,
#         map_location="cpu",
#         weights_only=True,
#     )


#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()




    

# if __name__ == "__main__":
#     main()