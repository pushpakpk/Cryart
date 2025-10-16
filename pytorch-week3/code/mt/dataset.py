import torch
from torch.utils.data import Dataset, DataLoader
import random

class ToyTranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=10):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tokens = [self.src_vocab.get(token, self.src_vocab['<unk>']) for token in self.src_sentences[idx].split()]
        tgt_tokens = [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) for token in self.tgt_sentences[idx].split()]

        src_tokens = src_tokens[:self.max_len]
        tgt_tokens = tgt_tokens[:self.max_len]

        src_tokens = torch.tensor([self.src_vocab['<sos>']] + src_tokens + [self.src_vocab['<eos>']])
        tgt_tokens = torch.tensor([self.tgt_vocab['<sos>']] + tgt_tokens + [self.tgt_vocab['<eos>']])

        return src_tokens, tgt_tokens

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_batch, tgt_batch, src_lens, tgt_lens

def get_toy_loaders(batch_size=32):
    # Simple toy dataset
    src_sentences = ["i am a student", "you are a teacher", "he likes pizza", "she reads books"]*50
    tgt_sentences = ["je suis un etudiant", "tu es un professeur", "il aime la pizza", "elle lit des livres"]*50

    # Build vocab
    src_vocab = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
    tgt_vocab = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
    for sent in src_sentences:
        for tok in sent.split():
            if tok not in src_vocab: src_vocab[tok] = len(src_vocab)
    for sent in tgt_sentences:
        for tok in sent.split():
            if tok not in tgt_vocab: tgt_vocab[tok] = len(tgt_vocab)

    dataset = ToyTranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader, src_vocab, tgt_vocab
