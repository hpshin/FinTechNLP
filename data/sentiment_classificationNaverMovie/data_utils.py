import argparse
import numpy as np
import pandas as pd
import torch.utils.data
import konlpy
from konlpy.tag import Mecab

def tokenizer(string):
    string.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    mecab = Mecab()

    return mecab.morphs(string)

def apply_tokenizer(dataframe):
    print('[Info] Tokenize...')
    return dataframe.apply(tokenizer).tolist()

def build_vocab(sentence):
    print('[Info] Build vocabulary')
    vocab_set = set(token for sent in sentence for token in sent)

    vocab = { "<pad>": 0, "<unk>": 1}
    start_index = len(vocab)
    for i, token in enumerate(vocab_set):
        vocab[token] = start_index + i    
    return vocab
    
def vectorize(vocab, sentences, one_sentence=False):
    UNK = vocab.get('<unk>')
    if one_sentence:
        return [vocab.get(token, UNK) for token in sentences]
    return [[vocab.get(token, UNK) for token in sent] for sent in sentences]

def build_dataset(data_path, split_ratio=None, predefined_vocab=None, generate_bigrams=None):
    dataframe = pd.read_table(data_path, sep='\t')
    print('[Info] Get {} data from {}'.format(len(dataframe), data_path))
    dataframe = dataframe.dropna(how='any')
    print('[Info] Drop null data, now the length of this data is {}'.format(len(dataframe)))

    if split_ratio:
        dataframe = pd.DataFrame(np.random.permutation(dataframe), columns=['id', 'document', 'label']) 
    sentence, label = apply_tokenizer(dataframe['document']), dataframe['label'].tolist()

    def make_bigrams(x):
        for sent in x:
            n_grams = list(zip(*[sent[i:] for i in range(2)]))
            for n_gram in n_grams:
                sent.append(' '.join(n_gram))
        return x

    if generate_bigrams:
        sentence = make_bigrams(sentence)

    if predefined_vocab:
        print('[Info] Pre-defined vocabulary found.')
        vocab = predefined_vocab
    else:
        vocab = build_vocab(sentence)
    print('[Info] Vocabulary size=', len(vocab))
    vec_sentence = vectorize(vocab, sentence)
    
    if split_ratio:
        n_split = int(len(sentence) * split_ratio)
        trn_sentence, trn_label = vec_sentence[:n_split], label[:n_split]
        val_sentence, val_label = vec_sentence[n_split:], label[n_split:]
        print('[Info] Split {} data to {} for train data,  {} for valid data.'
            .format(len(vec_sentence), len(trn_sentence), len(val_sentence)))

        trn_dataset, val_dataset = MovieDataset(vocab, trn_sentence, trn_label), MovieDataset(vocab, val_sentence, val_label)
        print('[Info] Build datasets')
        return trn_dataset, val_dataset
    else:
        dataset = MovieDataset(vocab, vec_sentence, label)
        print('[Info] Build dataset')
        return dataset 

def padding(inputs): 
    sentence, label = list(zip(*inputs))
    sentence = torch.nn.utils.rnn.pad_sequence(sentence, batch_first=True, padding_value=0)
    batch =  [ sentence, torch.cat(label, dim=0) ]
    return batch

def build_dataloader(dataset, batch_size, collate_fn=padding):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader

class MovieDataset(torch.utils.data.Dataset):
    """ x: sentence1, sentence2, y: gold_label """
    def __init__(self, vocab, sentence, label):
        self.vocab = vocab
        self.sentence = sentence
        self.label = label

    def __len__(self):
        return len(self.sentence)
    
    def __getitem__(self, idx):
        sentence = torch.LongTensor(self.sentence[idx])
        label = torch.LongTensor([self.label[idx]])
        return sentence, label