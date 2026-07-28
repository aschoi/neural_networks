import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time

from model_reverseEnglish.transformer_model import Transformer
from model_reverseEnglish.data_utils import create_synthetic_data, Vocabulary, TranslationDataset
from model_reverseEnglish.train import TransformerTrainer
from model_reverseEnglish.inference import TransformerInference
from datasets import load_dataset

from itertools import chain
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def beam_search(self, model, src_vocab, tgt_vocab, src_sentence, beam_width=3, max_len=50):
        """
        Beam Search Decoding - maintains multiple hypothesis
        
        Args:
            src_sentence:   <type>  
            beam_width:     <int>     
            max_len:        <int>   
        Return:
            self.tgt_vocab.decode(best_beam['tokens'])      <type>
        """
        # Encode source w/ consistent max_len
        src_tokens = torch.tensor([self.src_vocab.encode(src_sentence, max_len=15)], dtype=torch.int64)
        src_mask = self.model.create_padding_mask(src_tokens)

        # Initialize Beam w/ SOS token
        beams = [{'tokens': [self.tgt_vocab.token2Idx['[BOS]']], 'score': 0.0}]


        for step in range(max_len):
            candidates = []

            for beam in beams:
                if beam['tokens'][-1] == self.tgt_vocab.token2Idx['[EOS]']:
                    candidates.append(beam)
                    continue

                # Get Cur Seq
                tgt_tokens = torch.tensor([beam['tokens']], dtype=torch.int64)

                tgt_causal_mask = self.model.create_causal_mask(tgt_tokens.size(1))
                tgt_padding_mask = self.model.create_padding_mask(tgt_tokens)
                tgt_mask = tgt_causal_mask & tgt_padding_mask

                # Forward Pass
                with torch.no_grad():
                    output = self.model(src_tokens, tgt_tokens, src_mask, tgt_mask)

                # Get Probabilities for next token
                logits = output[:, -1, :]
                probs = F.log_softmax(logits, dim=-1)

                # Get top-k candidates
                top_probs, top_indices = torch.topk(probs, beam_width)

                for prob, idx in zip(top_probs[0], top_indices[0]):
                    new_tokens = beam['tokens'] + [idx.item()]
                    new_score = beam['score'] + prob.item()
                    candidates.append({'tokens': new_tokens, 'score': new_score})

            # Select top beam_width candidates
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = candidates[:beam_width]

            # Check if all beams ended
            if all(beam['tokens'][-1] == self.tgt_vocab.token2Idx['<EOS>'] for beam in beams):
                break

        best_beam = max(beams, key=lambda x: x['score'])
        return self.tgt_vocab.decode(best_beam['tokens'])


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

    def collate_fn(batch):
        """
        Custom collate function
        Args:

        Return:
        """
        src_seqs = []
        tgt_seqs = []

        for sample in batch:
            src_ids = tokenizer.encode(sample['en']).ids
            tgt_ids = tokenizer.encode(sample['de']).ids
            src = torch.tensor([BOS_ID, *src_ids, EOS_ID], dtype=torch.int64)
            tgt = torch.tensor([BOS_ID, *tgt_ids, EOS_ID], dtype=torch.int64)

            src_seqs.append(src)
            tgt_seqs.append(tgt)

        src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=PAD_ID)
        tgt_batch = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_ID)

        return {
            "src": src_batch,
            "tgt": tgt_batch[:, :-1],
            "tgt_output": tgt_batch[:, 1:],
            "source_padding_mask": src_batch.eq(PAD_ID),
            "target_padding_mask": tgt_batch[:, :-1].eq(PAD_ID),
        }


    print("Testing Transformer Inference.")

    dataset = load_dataset("bentrevett/multi30k")

    print(type(dataset))
    print(dataset["train"][0])
    print(dataset['train'])
    print(dataset)

    training_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    tokenizer = tokenize(training_dataset)

    PAD_ID = tokenizer.token_to_id("[PAD]")
    UNK_ID = tokenizer.token_to_id("[UNK]")
    BOS_ID = tokenizer.token_to_id("[BOS]")
    EOS_ID = tokenizer.token_to_id("[EOS]")

    # self.token2Idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}

    VOCAB_SIZE = tokenizer.get_vocab_size()  

    # # Prepare Data
    # src_sentences, tgt_sentences = create_synthetic_data(num_samples=200)  # list<string>
    # src_vocab = Vocabulary()        # dict{idx: token}, dict{token: idx}, size
    # tgt_vocab = Vocabulary()        # dict{idx: token}, dict{token: idx}, size
    # src_vocab.build_vocab(src_sentences)
    # tgt_vocab.build_vocab(tgt_sentences)

    # dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=15)
    train_loader = DataLoader(training_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # Create Model
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.1
    )

    # Train Model
    trainer = TransformerTrainer(model, train_loader, lr=1e-3, warmup_steps=25)
    epochs = 5
    print(f"\n-----Training for {epochs} epochs-----")
    for epoch in range(1, epochs+1):
        avg_loss = trainer.train_epoch()
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}\n")

    # # Test Model Inference
    # inference = TransformerInference(model, src_vocab, tgt_vocab)
    # test_sentences = ["hello world", "good morning", "thank you very much"]
    # beam_widths = [1, 3, 5]
    
    # print("\nInference Results:")
    # for w in beam_widths:
    #     print(f"\nBeam Width: {w}")
    #     for sentence in test_sentences:
    #         print(f"    Source: '{sentence}'")
    #         print(f"    Expected: '{' '.join(sentence.split()[::-1])}'")

    #         start_time = time.time()
    #         beam_result = inference.beam_search(sentence, beam_width=w)
    #         beam_time = time.time() - start_time
                    
    #         print(f"    Beam: '{beam_result}' (time: {beam_time:.3f}s)")
    #         print("    " + "-" * 50)


if __name__ == "__main__":
    main()


        


