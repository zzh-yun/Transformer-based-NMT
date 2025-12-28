import os
import re
import torch
import hydra
from omegaconf import DictConfig
from typing import Optional
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import pickle
import jieba
from .utils import get_data




class Vocabulary:
    """Vocabulary class for managing word-to-index mappings"""
    
    def __init__(self, cfgs):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_count = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3

        self.configs = cfgs
    
    def add_tokens(self, tokens):
        """Add a list of tokens to the vocabulary"""
        for token in tokens:
            self.word_count[token] += 1
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
    
    def build_vocab(self, list_of_token_lists, min_freq=1):
        """Build vocabulary from a list of token lists"""
        for tokens in list_of_token_lists:
            for token in tokens:
                self.word_count[token] += 1
        
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def tokens_to_indices(self, tokens):
        """Convert a list of tokens to indices"""
        return [self.word2idx.get(token, self.UNK_IDX) for token in tokens]

    def indices_to_tokens(self, indices):
        """Convert indices back to tokens"""
        tokens = []
        for idx in indices:
            if idx == self.EOS_IDX:
                break
            if idx != self.PAD_IDX and idx != self.SOS_IDX:
                tokens.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return tokens
    
    def indices_to_sentence(self, indices):
        """Convert indices back to sentence string"""
        tokens = self.indices_to_tokens(indices)
        return " ".join(tokens)
    
    def decode_batch(self, batch_indices):
        """Decode a batch of indices to sentences"""
        sentences = []
        for indices in batch_indices:
            tokens = self.indices_to_tokens(indices)
            sentences.append(" ".join(tokens))
        return sentences

    def __len__(self):
        return len(self.word2idx)


class NMTDataset(Dataset):
    """Dataset class for Neural Machine Translation"""
    
    def __init__(self, src_token_lists, tgt_token_lists, src_vocab, tgt_vocab, max_len=128):
        self.src_token_lists = src_token_lists
        self.tgt_token_lists = tgt_token_lists
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len  # 用于控制一个batch中句子的长度(or 在这里便是控制数据集中所有句子的长度) 但是只有截断，并没有补全的功能
    
    def __len__(self):
        return len(self.src_token_lists)
    
    def __getitem__(self, idx):
        src_tokens = self.src_token_lists[idx]
        tgt_tokens = self.tgt_token_lists[idx]
# 
        src_indices = self.src_vocab.tokens_to_indices(src_tokens)[:self.max_len]

        tgt_input_tokens = ['<SOS>'] + tgt_tokens
        tgt_output_tokens = tgt_tokens+ ['<EOS>']

        tgt_input_indices = self.tgt_vocab.tokens_to_indices(tgt_input_tokens)[:self.max_len]
        tgt_output_indices = self.tgt_vocab.tokens_to_indices(tgt_output_tokens)[:self.max_len]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input_indices, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output_indices, dtype=torch.long)
        }


class DataPreprocessor:
    """Data preprocessing class for NiuTrans dataset"""
    
    def __init__(self, cfgs):
        self.data_dir = cfgs.DATA_DIR
        self.configs = cfgs
        self.src_vocab = Vocabulary(cfgs)
        self.tgt_vocab = Vocabulary(cfgs)

    
    def create_datasets(self, data_dir):
        """Create train and validation datasets"""
        # Assuming en-zh translation for now
        # TODO: set the correct data path
        # src_sents, tgt_sents = self.load_parallel_data('english.raw.txt', 'chinese.raw.txt')   # 设置正确的目录
        (src_train_sents, tgt_train_sents,
            src_valid_sents, tgt_valid_sents,
            src_test_sents, tgt_test_sents) = get_data(data_dir)

        # Build vocabularies from training data only
        self.src_vocab.build_vocab(src_train_sents, min_freq=self.configs.min_freq)
        self.tgt_vocab.build_vocab(tgt_train_sents, min_freq=self.configs.min_freq)
        print(f"Source vocabulary size: {len(self.src_vocab)}")
        print(f"Target vocabulary size: {len(self.tgt_vocab)}")
        
        # Split data
        # num_train = int(len(src_sents) * train_ratio)
        # train_src = src_sents[:num_train]
        # train_tgt = tgt_sents[:num_train]
        # val_src = src_sents[num_train:]
        # val_tgt = tgt_sents[num_train:]
        train_src = src_train_sents
        train_tgt = tgt_train_sents
        val_src = src_valid_sents
        val_tgt = tgt_valid_sents
        test_src = src_test_sents
        test_tgt = tgt_test_sents
        
        train_dataset = NMTDataset(train_src, train_tgt, self.src_vocab, self.tgt_vocab, max_len=self.configs.MAX_LEN)
        val_dataset = NMTDataset(val_src, val_tgt, self.src_vocab, self.tgt_vocab, max_len=self.configs.MAX_LEN)
        test_dataset = NMTDataset(test_src, test_tgt, self.src_vocab, self.tgt_vocab, max_len=self.configs.MAX_LEN)
        
        return train_dataset, val_dataset, test_dataset

    def save_vocabularies(self, save_dir='checkpoints'):
        """Save source and target vocabularies to pickle files"""
        os.makedirs(save_dir, exist_ok=True)
        src_path = os.path.join(save_dir, 'src_vocab.pkl')
        tgt_path = os.path.join(save_dir, 'tgt_vocab.pkl')
        
        with open(src_path, 'wb') as f:
            pickle.dump(self.src_vocab, f)
        
        with open(tgt_path, 'wb') as f:
            pickle.dump(self.tgt_vocab, f)
            
        print(f"Vocabularies saved to {save_dir}")

    def load_vocabularies(self, save_dir='checkpoints'):
        """Load vocabularies from pickle files"""
        src_path = os.path.join(save_dir, 'src_vocab.pkl')
        tgt_path = os.path.join(save_dir, 'tgt_vocab.pkl')
        
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            with open(src_path, 'rb') as f:
                self.src_vocab = pickle.load(f)
            with open(tgt_path, 'rb') as f:
                self.tgt_vocab = pickle.load(f)
            print("Vocabularies loaded successfully.")
            return True
        else:
            print("Vocabulary files not found, building from scratch.")
            return False


def collate_fn(batch):
    """Custom collate function to pad sequences in a batch"""
    src_batch, tgt_input_batch, tgt_output_batch = [], [], []
    
    pad_idx = batch[0]['src_vocab'].PAD_IDX if 'src_vocab' in batch[0] else 0

    for item in batch:
        src_batch.append(item['src'])
        tgt_input_batch.append(item['tgt_input'])
        tgt_output_batch.append(item['tgt_output'])
        
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_idx, batch_first=False)
    tgt_input_batch = torch.nn.utils.rnn.pad_sequence(tgt_input_batch, padding_value=pad_idx, batch_first=False)
    tgt_output_batch = torch.nn.utils.rnn.pad_sequence(tgt_output_batch, padding_value=pad_idx, batch_first=False)

    return {
        'src': src_batch,
        'tgt_input': tgt_input_batch,
        'tgt_output': tgt_output_batch
    }


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4, distributed=False):
    """Create data loaders for training and validation"""
    # 当使用DistributedSampler时，不能同时设置shuffle=True
    # DistributedSampler本身可以通过shuffle参数控制是否打乱
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None
    # 用torch自带的dataloader不太好吧，没有对齐功能，或者说，在输入的train_dataset的时候，就要做对齐处理，Datasets只是提取它
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(not distributed),  # 只有在非分布式训练时才使用shuffle
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=train_sampler,  # 分布式训练时使用sampler，此时shuffle必须为False
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=val_sampler,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=test_sampler,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

@hydra.main(version_base='1.3', config_path='./configs', config_name='train.yaml')
def demo(cfgs: DictConfig) -> Optional[float]:
    # Test DataPreprocessor
    preprocessor = DataPreprocessor(cfgs)
    train_ds, val_ds = preprocessor.create_datasets()
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    
    # Test DataLoader
    train_dl, val_dl = create_data_loaders(train_ds, val_ds, batch_size=4)
    
    for batch in train_dl:
        print("Sample batch:")
        print("Source shape:", batch['src'].shape)
        print("Tgt Input shape:", batch['tgt_input'].shape)
        print("Tgt Output shape:", batch['tgt_output'].shape)
        
        # Test decoding
        print("Sample src:", preprocessor.src_vocab.decode_batch(batch['src'].transpose(0,1).numpy())[0])
        print("Sample tgt:", preprocessor.tgt_vocab.decode_batch(batch['tgt_output'].transpose(0,1).numpy())[0])
        break 


if __name__ == '__main__':
    demo()