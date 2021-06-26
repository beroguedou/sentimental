import re
import torch
import pandas as pd
import transformers
from config import *

    
class SentimentalDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, max_length, sentences, labels):
        'Initialization'
        self.max_length = max_length
        self.labels = labels
        self.sentences = sentences

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sentences)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        text = self.sentences[index]
        label = self.labels[index]

        return text, label
    
    
class SentimentalInfDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, max_length, sentences_ids, sentences):
        'Initialization'
        self.max_length = max_length
        self.sentences_ids = sentences_ids
        self.sentences = sentences

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sentences)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        text = self.sentences[index]
        id_text = self.sentences_ids[index]

        return text, id_text
    
def text_cleaner(text):
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Converting to Lowercase
    text = text.lower()
    return text
    
    
def train_collate_fn(batch):
    # soted the shortest to the longest sentence
    batch = sorted(batch, key= lambda x: len(x[0]))
    data = [item[0] for item in batch]
    data = torch.LongTensor(tokenizer(data,
                           padding='longest',
                           max_length=max_length, 
                           truncation=True)['input_ids'])
    
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def batch_inf_collate_fn(batch):
    # soted the shortest to the longest sentence
    batch = sorted(batch, key= lambda x: len(x[0]))
    data = [item[0] for item in batch]
    data = torch.LongTensor(tokenizer(data,
                           padding='longest',
                           max_length=max_length, 
                           truncation=True)['input_ids'])
    
    ids = [item[1] for item in batch]
    return [data, ids]


class SentimentalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(SentimentalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = train_collate_fn
        
        
class SentimentalInfDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(SentimentalInfDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = batch_inf_collate_fn