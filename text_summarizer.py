import spacy
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

spacy_en = spacy.load('en_core_web_sm')

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def tokenize_en(text):
    """Tokenizes English text using spaCy."""
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_data(src, tgt, lowercase=False):
    """Loads source and target data, tokenizes, and returns paired data."""
    with open(src, 'r', encoding='utf-8') as src_file, open(tgt, 'r', encoding='utf-8') as tgt_file:
        src_lines = [line.strip().lower() if lowercase else line.strip() for line in src_file.readlines()]
        tgt_lines = [line.strip().lower() if lowercase else line.strip() for line in tgt_file.readlines()]

    print(f"LOADED: {len(src_lines)} source lines, {len(tgt_lines)} target lines")
    assert len(src_lines) == len(tgt_lines), "Mismatch between source and target lines!"

    return list(zip(src_lines, tgt_lines))

class SummarizationDataset(Dataset):
    """Custom Dataset for handling summarization data."""
    def __init__(self, data, tokenizer, max_src_len=512, max_tgt_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_tokens = self.tokenizer(src)[:self.max_src_len]
        tgt_tokens = self.tokenizer(tgt)[:self.max_tgt_len]
        return {
            "src": src_tokens,
            "tgt": tgt_tokens
        }

def collate_fn(batch, pad_token="<pad>"):
    """Pads source and target sequences in the batch to the same length."""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]

    src_max_len = max(len(seq) for seq in src_batch)
    tgt_max_len = max(len(seq) for seq in tgt_batch)

    def pad_sequence(seq, max_len):
        return seq + [pad_token] * (max_len - len(seq))

    src_padded = [pad_sequence(seq, src_max_len) for seq in src_batch]
    tgt_padded = [pad_sequence(seq, tgt_max_len) for seq in tgt_batch]

    return {
        "src": src_padded,
        "tgt": tgt_padded
    }

# Load Data
data_paths = {
    "train": ("cnndm/data/train.txt.src", "cnndm/data/train.txt.tgt"),
    "dev": ("cnndm/data/val.txt.src", "cnndm/data/val.txt.tgt"),
    "test": ("cnndm/data/test.txt.src", "cnndm/data/test.txt.tgt")
}

train_data = load_data(*data_paths["train"], lowercase=True)
dev_data = load_data(*data_paths["dev"], lowercase=True)
test_data = load_data(*data_paths["test"], lowercase=True)

# Create Dataset Instances
train_dataset = SummarizationDataset(train_data, tokenize_en)
dev_dataset = SummarizationDataset(dev_data, tokenize_en)
test_dataset = SummarizationDataset(test_data, tokenize_en)

# Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
