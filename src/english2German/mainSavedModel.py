import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time

from .model import Transformer
from .train import TransformerTrainer
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from collections.abc import Iterable
from pathlib import Path



def main():

    def collate_fn(batch):
        """
        Custom collate function
        Args:

        Return:
        """
        src_seqs = []
        tgt_seqs = []
  
        englishTexts = []
        germanTexts = []

        for sample in batch:
            engSample = sample["en"]
            englishTexts.append(engSample)
            bos_engEncodings_eos = encode_source(engSample)
            bos_engEncodings_eos_tensor = torch.tensor(bos_engEncodings_eos, dtype=torch.long)
            src_seqs.append(bos_engEncodings_eos_tensor)

            germSample = sample["de"]
            germanTexts.append(germSample)
            bos_deEncodings_eos = encode_target(germSample)
            bos_deEncodings_eos_tensor = torch.tensor(bos_deEncodings_eos, dtype=torch.long)
            tgt_seqs.append(bos_deEncodings_eos_tensor)

        t_bos_srcEncodings_eos_pads_asIds = pad_sequence(src_seqs, batch_first=True, padding_value=SRC_PAD_ID)
        t_bos_tgtEncodings_eos_pads_asIds = pad_sequence(tgt_seqs, batch_first=True, padding_value=TGT_PAD_ID)

        # [BOS, token1, token2, ..., tokenN]
        t_bos_tgtEncodings_pads_asIds = t_bos_tgtEncodings_eos_pads_asIds[:, :-1]

        # [token1, token2, ..., tokenN, EOS]
        t_tgtEncodings_eos_pads_asIds = t_bos_tgtEncodings_eos_pads_asIds[:, 1:]

        return {
            "bos_src_eos": t_bos_srcEncodings_eos_pads_asIds,
            "bos_tgt": t_bos_tgtEncodings_pads_asIds,
            "tgt_eos": t_tgtEncodings_eos_pads_asIds,
            "bos_tgt_eos": t_bos_tgtEncodings_eos_pads_asIds,
            "en": englishTexts,
            "de": germanTexts
        }

    def encode_source(text: str) -> list[int]:
        encoding = source_tokenizer.encode(text)

        return [SRC_BOS_ID, *encoding.ids, SRC_EOS_ID,]


    def encode_target(text: str) -> list[int]:
        encoding = target_tokenizer.encode(text)

        return [TGT_BOS_ID, *encoding.ids, TGT_EOS_ID,]


    SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
    unk_token = '[UNK]'

    # ----- Main -------
    print("Continue Training From Saved Model.")

    dataset = load_dataset("bentrevett/multi30k")

    training_dataset = dataset['train']
    validation_dataset = dataset['validation']
    #  test_dataset = dataset['test']

    source_tokenizer = Tokenizer.from_file("tokenizers/english_bpe.json")
    target_tokenizer = Tokenizer.from_file("tokenizers/german_bpe.json")

    SRC_PAD_ID = source_tokenizer.token_to_id("[PAD]")
    SRC_UNK_ID = source_tokenizer.token_to_id("[UNK]")
    SRC_BOS_ID = source_tokenizer.token_to_id("[BOS]")
    SRC_EOS_ID = source_tokenizer.token_to_id("[EOS]")

    TGT_PAD_ID = target_tokenizer.token_to_id("[PAD]")
    TGT_UNK_ID = target_tokenizer.token_to_id("[UNK]")
    TGT_BOS_ID = target_tokenizer.token_to_id("[BOS]")
    TGT_EOS_ID = target_tokenizer.token_to_id("[EOS]")

    source_vocab_size = source_tokenizer.get_vocab_size()
    target_vocab_size = target_tokenizer.get_vocab_size()

    train_loader = DataLoader(training_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset,batch_size=64, shuffle=True,collate_fn=collate_fn)
    
    # Create Model
    model = Transformer(
        src_vocab_size=source_vocab_size,
        tgt_vocab_size=target_vocab_size,
        src_pad_id=SRC_PAD_ID,
        tgt_pad_id=TGT_PAD_ID,
        d_model=256,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=1024,
        dropout=0.2,
        activation='gelu'
    )

    parameter_name = next(iter(model.state_dict()))
    before = model.state_dict()[parameter_name].clone()
    
    checkpoint_path = Path("checkpoints/english2German_latest.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    startEpoch = checkpoint['epoch']
    
    # Train Model
    trainer = TransformerTrainer(model, train_loader, validation_loader, TGT_PAD_ID, lr=3e-4, warmup_steps=800)

    start = time.perf_counter()

    epochs = 30
    print(f"\n-----Training-----")
    print(f'saved state epoch count: {startEpoch}, training for {epochs} epochs')
    for curEpoch in range(startEpoch, epochs+1):
        training_loss, validation_loss = trainer.train_epoch(curEpoch)
        print(f"Epoch {curEpoch}, Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss}\n")


if __name__ == "__main__":
    main()


        


