import spacy

spacy_en = spacy.load('en_core_web_sm')  # Disabling unnecessary components

def tokenize_en(text):
    """Tokenizes English text using spaCy."""
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_data(src, tgt):
    """Loads data and tokenizes it efficiently using spaCy's pipe."""
    with open(src, 'r', encoding='utf-8') as src_file, open(tgt, 'r', encoding='utf-8') as tgt_file:
        src_lines = [line.strip() for line in src_file.readlines()]
        tgt_lines = [line.strip() for line in tgt_file.readlines()]
    
    assert len(src_lines) == len(tgt_lines), "Mismatch between source and target lines!"

    # ⚡ Use spaCy's `nlp.pipe()` for faster batch tokenization
    src_tokenized = [[tok.text for tok in doc] for doc in spacy_en.pipe(src_lines, batch_size=100)]
    tgt_tokenized = [[tok.text for tok in doc] for doc in spacy_en.pipe(tgt_lines, batch_size=100)]

    return list(zip(src_tokenized, tgt_tokenized))

# Example usage:
train_data = load_data("cnndm/data/train.txt.src", "cnndm/data/train.txt.tgt")
print(train_data[:1])  # Print first 1 example
