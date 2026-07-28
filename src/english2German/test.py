import torch
import torch.nn as nn
import torch.optim as optim
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

from dataclasses import dataclass
import sacrebleu
from pathlib import Path

@dataclass
class TestResults:
    bleu: float
    exact_match_percent: float
    token_accuracy_percent: float
    average_test_loss: float
    perplexity: float
    predictions: list[str]
    references: list[str]


def greedy_decode_batch(
    model: nn.Module,
    source: torch.Tensor,
    sos_index: int,
    eos_index: int,
    max_output_length: int,
) -> torch.Tensor:
    """
    Generate target sequences autoregressively.

    Args:
        source:
            [batch_size, source_length]

    Returns:
        generated:
            [batch_size, generated_length]
    """

    batch_size = source.size(0)

    generated = torch.full(size=(batch_size, 1), fill_value=sos_index, dtype=torch.long)
    finished = torch.zeros(batch_size, dtype=torch.bool)

    for _ in range(max_output_length):
        logits = model(source, generated)

        # Last output position:
        # [batch_size, target_vocab_size]
        next_token_logits = logits[:, -1, :]

        next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1 )

        finished |= next_token.squeeze(1).eq(eos_index)

        if finished.all():
            break

    return generated



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



def main():

    # def collate_fn(batch):
    #     """
    #     Custom collate function
    #     Args:

    #     Return:
    #     """
    #     src_seqs = []
    #     tgt_seqs = []

    #     for sample in batch:
    #         src_ids = tokenizer.encode(sample['en']).ids
    #         tgt_ids = tokenizer.encode(sample['de']).ids
    #         src = torch.tensor([BOS_ID, *src_ids, EOS_ID], dtype=torch.int64)
    #         tgt = torch.tensor([BOS_ID, *tgt_ids, EOS_ID], dtype=torch.int64)

    #         src_seqs.append(src)
    #         tgt_seqs.append(tgt)

    #     src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=PAD_ID)
    #     tgt_batch = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_ID)

    #     return {
    #         "src": src_batch,
    #         "tgt": tgt_batch[:, :-1],
    #         "tgt_output": tgt_batch[:, 1:],
    #         "src_padding_mask": src_batch.eq(PAD_ID),
    #         "tgt_padding_mask": tgt_batch[:, :-1].eq(PAD_ID),
    #     }
    def collate_fn(batch):
        src_sequences = []
        full_tgt_sequences = []

        for example in batch:
            src_ids = tokenizer.encode(example["en"]).ids
            tgt_ids = tokenizer.encode(example["de"]).ids
            src = torch.tensor([BOS_ID, *src_ids, EOS_ID], dtype=torch.int64)
            tgt = torch.tensor([BOS_ID, *tgt_ids, EOS_ID], dtype=torch.int64)

            src_sequences.append(src)
            full_tgt_sequences.append(tgt)

        src = pad_sequence(src_sequences, batch_first=True, padding_value=PAD_ID)

        full_tgt = pad_sequence(full_tgt_sequences, batch_first=True, padding_value=PAD_ID)

        # Example full target:
        # [BOS, ein, mann, läuft, EOS]
        #
        # Decoder input:
        # [BOS, ein, mann, läuft]
        tgt_input = full_tgt[:, :-1]

        # Expected model output:
        # [ein, mann, läuft, EOS]
        tgt_output = full_tgt[:, 1:]

        return {
            "src": src,
            "tgt": tgt_input,
            "tgt_output": tgt_output,
            "target_full": full_tgt,
        }

    print("Testing Model against BLEU.")

    dataset = load_dataset("bentrevett/multi30k")

    training_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    tokenizer = tokenize(training_dataset)

    PAD_ID = tokenizer.token_to_id("[PAD]")
    UNK_ID = tokenizer.token_to_id("[UNK]")
    BOS_ID = tokenizer.token_to_id("[BOS]")
    EOS_ID = tokenizer.token_to_id("[EOS]")

    VOCAB_SIZE = tokenizer.get_vocab_size()  

    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    
    # Create Model
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        dropout=0.1
    )

    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)


    checkpoint_path = Path("checkpoints/english2German_latest.pt")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    predictions: list[str] = []
    references: list[str] = []

    total_loss = 0.0
    total_loss_tokens = 0

    total_correct_tokens = 0
    total_compared_tokens = 0

    with torch.inference_mode():


        for batch in test_loader:
            # Parse Batch Data
            source = batch['src']
            target = batch['target_full']

            # --------------------------------------------
            # 1. Teacher-forced test loss
            # --------------------------------------------

            decoder_input = target[:, :-1]
            expected_output = target[:, 1:]

            logits = model(
                source,
                decoder_input,
            )

            vocabulary_size = logits.size(-1)

            loss = criterion(
                logits.reshape(-1, vocabulary_size),
                expected_output.reshape(-1),
            )

            non_pad_count = (
                expected_output != PAD_ID
            ).sum().item()

            total_loss += loss.item() * non_pad_count
            total_loss_tokens += non_pad_count

            # --------------------------------------------
            # 2. Autoregressive generation
            # --------------------------------------------

            generated = greedy_decode_batch(
                model=model,
                source=source,
                sos_index=BOS_ID,
                eos_index=EOS_ID,
                max_output_length=100,
            )

            # --------------------------------------------
            # 3. Convert generated/reference IDs to text
            # --------------------------------------------

            for predicted_ids, reference_ids in zip(
                generated.tolist(),
                target.tolist(),
            ):


                predicted_text = tokenizer.decode(
                    predicted_ids,
                    skip_special_tokens=True,
                )

                reference_text = tokenizer.decode(
                    reference_ids,
                    skip_special_tokens=True,
                )

                predictions.append(predicted_text)
                references.append(reference_text)


                # Token accuracy compares positions up to the
                # longer of the two sequences.
                comparison_length = max(
                    len(predicted_text),
                    len(reference_text),
                )

                total_compared_tokens += comparison_length

                for index in range(comparison_length):
                    predicted_token = (
                        predicted_text[index]
                        if index < len(predicted_text)
                        else None
                    )

                    reference_token = (
                        reference_text[index]
                        if index < len(reference_text)
                        else None
                    )

                    if predicted_token == reference_token:
                        total_correct_tokens += 1

    if not references:
        raise ValueError("The test loader produced no examples.")

    if total_loss_tokens == 0:
        raise ValueError(
            "The test set contained no non-padding target tokens."
        )

    # SacreBLEU expects:
    # predictions: list[str]
    # references:  list[list[str]]
    #
    # Each inner list is one complete reference corpus.
    bleu_result = sacrebleu.corpus_bleu(
        predictions,
        [references],
    )

    exact_matches = sum(
        prediction.strip() == reference.strip()
        for prediction, reference in zip(
            predictions,
            references,
        )
    )

    exact_match_percent = (
        100.0 * exact_matches / len(references)
    )

    token_accuracy_percent = (
        100.0 * total_correct_tokens / total_compared_tokens
        if total_compared_tokens > 0
        else 0.0
    )

    average_test_loss = total_loss / total_loss_tokens

    # Prevent overflow for a severely undertrained model.
    perplexity = (
        torch.exp(
            torch.tensor(
                min(average_test_loss, 100.0)
            )
        ).item()
    )

    print(f'bleu:  {bleu_result.score}')
    print(f'exact_match_percent:  {exact_match_percent}')
    print(f'token_accuracy_percent:  {token_accuracy_percent}')
    print(f'average_test_loss:  {average_test_loss}')
    print(f'perplexity:  {perplexity}')
    print(f'predictions:  {predictions}')
    print(f'references:  {references}')


    for index in range(min(10, len(predictions))):
        print(f"\nExample {index + 1}")
        print(f"Predicted: {predictions[index]}")
        print(f"Reference: {references[index]}")




if __name__ == "__main__":
    main()