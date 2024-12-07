import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from ydata_profiling import ProfileReport
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Sampler
import os
import json


class PrepareData:
    def __init__(self):
        self.word_eng = None
        self.word_darija = None
        self.eng_vocab = None
        self.darija_vocab = None

    def load_and_clean_data(self, dataset: str, folder: str, drop_column: str, output_csv: str) -> pd.DataFrame:
        # Load dataset
        print("Loading dataset...")
        df = load_dataset(dataset, folder)
        df = df[folder].to_pandas()

        # Drop unnecessary or invalid data
        df = df.dropna().drop_duplicates()
        if drop_column in df.columns:
            df.drop(drop_column, axis='columns', inplace=True)

        # Save cleaned data for debugging
        print(f"Data cleaned and saved to {output_csv}.")
        df.to_csv(output_csv, index=False)
        return df

    def clean_punctuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean punctuation from English and Darija sentences."""

        def clean_text(text: str) -> str:
            # Clean punctuation while preserving numbers
            cleaned_words = []
            for word in text.split():
                # Strip punctuation from start and end of word
                word = word.strip('.,!?()[]{}":;')
                if word:  # Only add non-empty words
                    cleaned_words.append(word)
            return ' '.join(cleaned_words)

        # Create a copy of the dataframe to avoid modifying the original
        cleaned_df = df.copy()

        # Clean both English and Darija text
        cleaned_df['eng'] = df['eng'].apply(clean_text)
        cleaned_df['darija'] = df['darija'].apply(clean_text)

        return cleaned_df

    def clean_text(self, text: str) -> str:
        """Remove unwanted characters and standardize text."""
        text = text.lower().strip()
        allowed_chars = set('abcdefghijklmnopqrstuvwxyz0123456789 ' + '3789')
        text = ''.join(c for c in text if c in allowed_chars)
        return ' '.join(text.split())

    def prepare_vocabularies(self, df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Create vocabularies for English and Darija, keeping all words."""
        eng_word_counts = {}
        darija_word_counts = {}

        # Count word frequencies (we will keep all words, so just count them)
        for sentence in df['eng']:
            for word in self.clean_text(sentence).split():
                eng_word_counts[word] = eng_word_counts.get(word, 0) + 1
        for sentence in df['darija']:
            for word in self.clean_text(sentence).split():
                darija_word_counts[word] = darija_word_counts.get(word, 0) + 1

        # No filtering by frequency, keep all words
        self.eng_vocab = set(eng_word_counts.keys())
        self.darija_vocab = set(darija_word_counts.keys())

        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        word_eng = {token: idx for idx, token in enumerate(special_tokens)}
        word_darija = {token: idx for idx, token in enumerate(special_tokens)}

        # Extend vocabularies with all words
        word_eng.update({word: idx for idx, word in enumerate(self.eng_vocab, start=len(word_eng))})
        word_darija.update({word: idx for idx, word in enumerate(self.darija_vocab, start=len(word_darija))})

        self.word_eng = word_eng
        self.word_darija = word_darija

        # Save vocabularies
        os.makedirs('vocabularies', exist_ok=True)
        with open('vocabularies/eng_vocab.json', 'w', encoding='utf-8') as f:
            json.dump(word_eng, f, ensure_ascii=False, indent=4)
        with open('vocabularies/darija_vocab.json', 'w', encoding='utf-8') as f:
            json.dump(word_darija, f, ensure_ascii=False, indent=4)

        print("Vocabularies prepared and saved.")
        print(f"English vocabulary size: {len(word_eng)}")
        print(f"Darija vocabulary size: {len(word_darija)}")
        return word_eng, word_darija

    def print_vocab_stats(self):
        """Print statistics about the vocabularies."""
        if self.eng_vocab is None or self.darija_vocab is None:
            raise ValueError("Vocabularies not created yet. Call prepare_vocabularies first.")

        print("\nVocabulary Statistics:")
        print("-" * 50)
        print(f"English vocabulary size: {len(self.word_eng)}")
        print(f"Darija vocabulary size: {len(self.word_darija)}")

        # Print some example words (excluding special tokens)
        print("\nExample English words:",
              list(word for word in list(self.eng_vocab)[:5]
                   if word not in ['<PAD>', '<UNK>', '<START>', '<END>']))
        print("Example Darija words:",
              list(word for word in list(self.darija_vocab)[:5]
                   if word not in ['<PAD>', '<UNK>', '<START>', '<END>']))

        # Print special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        print("\nSpecial tokens and their indices:")
        for token in special_tokens:
            print(f"English - {token}: {self.word_eng[token]}")
            print(f"Darija - {token}: {self.word_darija[token]}")

    def prepare_sentence(self, sentence: str, word_dict: Dict[str, int]) -> torch.Tensor:
        """Convert sentence to tensor of indices."""
        tokens = ['<START>'] + [
            word if word in word_dict else '<UNK>' for word in self.clean_text(sentence).split()
        ] + ['<END>']
        return torch.tensor([word_dict[token] for token in tokens], dtype=torch.long)

    def prepare_data_splits(self, df: pd.DataFrame, test_size: float = 0.3) -> Tuple:
        """Split data into training, validation, and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            df["eng"], df["darija"], test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42
        )

        print(f"Data splits: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def validate_vocab_and_data(self, df: pd.DataFrame):
        """Validate vocabulary and tokenization process."""
        sample_sentence = df['eng'].iloc[0]
        print(f"Sample Sentence: {sample_sentence}")
        sample_tensor = self.prepare_sentence(sample_sentence, self.word_eng)
        print(f"Tokenized Sentence: {sample_tensor.tolist()}")

    def print_length_stats(self, df: pd.DataFrame):
        """Print statistics about sentence lengths."""
        eng_lengths = df['eng'].str.split().str.len()
        darija_lengths = df['darija'].str.split().str.len()

        print("Length Statistics:")
        print(f"English - Mean: {eng_lengths.mean():.2f}, Max: {eng_lengths.max()}")
        print(f"Darija - Mean: {darija_lengths.mean():.2f}, Max: {darija_lengths.max()}")


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, word_eng, word_darija, max_length=50):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.word_eng = word_eng
        self.word_darija = word_darija
        self.max_length = max_length

    def truncate_and_pad_sequence(self, tokens, vocab, max_length):
        """Truncate sequence to max_length while preserving special tokens."""
        if len(tokens) > max_length:
            # Keep START token, truncate middle, add END token
            tokens = tokens[:1] + tokens[1:max_length - 1] + [tokens[-1]]

        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in vocab:
                indices.append(vocab[token])
            else:
                indices.append(vocab['<UNK>'])

        # Pad sequence if necessary
        if len(indices) < max_length:
            indices.extend([vocab['<PAD>']] * (max_length - len(indices)))

        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        # Get sentences
        eng_sentence = str(self.X.iloc[idx]).lower()
        darija_sentence = str(self.y.iloc[idx]).lower()

        # Tokenize with special tokens
        eng_tokens = ['<START>'] + eng_sentence.split() + ['<END>']
        darija_tokens = ['<START>'] + darija_sentence.split() + ['<END>']

        # Truncate and pad sequences
        eng_tensor = self.truncate_and_pad_sequence(eng_tokens, self.word_eng, self.max_length)
        darija_tensor = self.truncate_and_pad_sequence(darija_tokens, self.word_darija, self.max_length)

        # Calculate actual lengths (excluding padding)
        eng_length = (eng_tensor != self.word_eng['<PAD>']).sum().item()
        darija_length = (darija_tensor != self.word_darija['<PAD>']).sum().item()

        return {
            'eng': eng_tensor,
            'darija': darija_tensor,
            'eng_len': eng_length,
            'darija_len': darija_length
        }

    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    """Custom collate function for batching."""
    # Get max lengths in the batch
    max_eng_len = max(item['eng_len'] for item in batch)
    max_darija_len = max(item['darija_len'] for item in batch)

    # Prepare tensors
    batch_size = len(batch)
    eng_padded = torch.zeros((batch_size, max_eng_len), dtype=torch.long)
    darija_padded = torch.zeros((batch_size, max_darija_len), dtype=torch.long)

    # Fill tensors
    eng_lengths = []
    darija_lengths = []
    for i, item in enumerate(batch):
        eng_len = item['eng_len']
        darija_len = item['darija_len']

        eng_padded[i, :eng_len] = item['eng'][:eng_len]
        darija_padded[i, :darija_len] = item['darija'][:darija_len]

        eng_lengths.append(eng_len)
        darija_lengths.append(darija_len)

    return {
        'eng': eng_padded,
        'darija': darija_padded,
        'eng_lengths': torch.tensor(eng_lengths),
        'darija_lengths': torch.tensor(darija_lengths)
    }


# Updated data loader creation
def create_dataloaders(X_train, y_train, X_val, y_val, word_eng, word_darija, batch_size=32, max_length=50):
    """Create train and validation dataloaders with truncation."""
    # Create datasets with max_length parameter
    train_dataset = TranslationDataset(X_train, y_train, word_eng, word_darija, max_length=max_length)
    val_dataset = TranslationDataset(X_val, y_val, word_eng, word_darija, max_length=max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    return train_loader, val_loader
