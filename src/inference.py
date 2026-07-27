import torch
import torch.nn.functional as F


class TransformerInference:
    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Transformer Inference Constructor

        Args:
            model:      <type>
            src_vocab:  <type>
            tgt_vocab:  <type>
        """
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.model.eval()


    def greedy_decode(self, src_sentence, max_len=50):
        pass


    def beam_search(self, src_sentence, beam_width=3, max_len=50):
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
        src_tokens = torch.tensor(
            [self.src_vocab.encode(src_sentence, max_len=15)], dtype=torch.int64
        )
        src_mask = self.model.create_padding_mask(src_tokens)

        # Initialize Beam w/ SOS token
        beams = [{'tokens': [self.tgt_vocab.token2Idx['<SOS>']], 'score': 0.0}]


        for step in range(max_len):
            candidates = []

            for beam in beams:
                if beam['tokens'][-1] == self.tgt_vocab.token2Idx['<EOS>']:
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

