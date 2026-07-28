import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random


class Vocabulary:
    def __init__(self):
        """
        Vocabulary Module Constructor
        """
        self.token2Idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2Token = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.size = 4


    def add_token(self, token):
        """
        Args:
            token:   <string>    token to be added to dicts
        """
        if token not in self.token2Idx:
            self.token2Idx[token] = self.size
            self.idx2Token[self.size] = token
            self.size += 1


    def build_vocab(self, sentences, min_freq=1):
        """
        Build vocabulary from list of sentences

        Args:
            sentences:  <type>  
            min_freq:   <int>
        """
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence.split())

        for token, freq in counter.items():
            if freq >= min_freq:
                self.add_token(token)


    def encode(self, sentence, max_len=None):
        """
        Convert sentence to token indices

        Args:  
            sentence:   <string>    A sentence to be tokenized
        Return:
            indices     <list>
        """
        tokens = sentence.split()
        indices = [self.token2Idx.get(token, self.token2Idx['<UNK>']) for token in tokens]

        if max_len:
            indices = indices[:max_len-1]
            indices.append(self.token2Idx['<EOS>'])
            while len(indices) < max_len:
                indices.append(self.token2Idx['<PAD>'])
        
        return indices


    def decode(self, indices):
        """
        Convert token indices back to sentence

        Args:
            indices:    <type>  
        Return:
                        <string>
        """
        tokens = [self.idx2Token[idx] for idx in indices if idx != self.token2Idx['<PAD>']]

        return " ".join(tokens).replace('<SOS>', '').replace('<EOS>', '').strip()


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        """
        Args:
            src_sentences   <type>  
            tgt_sentences   <type>  
            src_vocab       <type>  
            tgt_vocab       <type>  
            max_len         <int>
        """
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len


    def __len__(self):
        return len(self.src_sentences)


    def __getitem__(self, idx):
        """
        Args:
            idx:    <int>  
        Return:
                    <dict{<string>: <tensor>}>  
        """
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]

        # Encode Sentences
        src_indices = self.src_vocab.encode(src_sentence, self.max_len)
        tgt_indices = self.tgt_vocab.encode('<SOS> ' + tgt_sentence, self.max_len)
        tgt_output = self.tgt_vocab.encode(tgt_sentence, self.max_len)

        return {
            'src': torch.tensor(src_indices, dtype=torch.int64),
            'tgt': torch.tensor(tgt_indices, dtype=torch.int64),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.int64)
        }


def create_synthetic_data(num_samples=1000):
    """
    Create synthetic translation data (English to reversed English)
    
    Args:
        num_samples         <int>           number of synthetic (repeated versions) samples to be created
    Return:
        src_sentences:      list<string>    a list containing the OG sentences. Each element is 1 sample sentence.         size: num_samples
        tgt_sentences:      list<string>    a list containing the "Translated" sentences.  Each elm is 1 sample sentence.  size: num_samples
    """
    templates = [
        "hello world", "good morning", "how are you", "nice to meet you", 
        "thank you very much", "see you later", "have a nice day"
    ]

    src_sentences = []
    tgt_sentences = []

    for sample in range(num_samples):
        template = random.choice(templates)
        src_sentences.append(template)
        tgt_sentences.append(" ".join(template.split()[::-1]))

    return src_sentences, tgt_sentences