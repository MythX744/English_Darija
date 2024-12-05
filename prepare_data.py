import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from ydata_profiling import ProfileReport
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class PrepareData:
    def __init__(self):
        self.eng_vocab = None
        self.darija_vocab = None
        self.word_eng = None
        self.word_darija = None

    def load_and_clean_data(self, dataset: str, folder: str,
                            drop_column: str, output_csv: str) -> pd.DataFrame:
        # Load dataset
        df = load_dataset(dataset, folder)
        df = df[folder].to_pandas()

        # Clean data
        df = df.dropna()
        df = df.drop_duplicates()
        if drop_column in df.columns:
            df.drop(drop_column, axis='columns', inplace=True)

        # Save cleaned data
        df.to_csv(output_csv, index=False)
        return df

    def create_profile_report(self, df: pd.DataFrame, title: str,
                              output_file: str = "profile.html") -> None:
        profile = ProfileReport(df, title=title, explorative=True)
        profile.to_file(output_file)

    def prepare_vocabularies(self, df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
        # Create vocabulary sets
        eng_vocab = set([word.lower() for sentence in df['eng']
                         for word in sentence.split()])
        darija_vocab = set([word.lower() for sentence in df['darija']
                            for word in sentence.split()])

        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']

        # Create word to index mappings
        word_eng = {token: idx for idx, token in enumerate(special_tokens)}
        word_darija = {token: idx for idx, token in enumerate(special_tokens)}

        # Add regular words
        for word in eng_vocab:
            if word not in word_eng:
                word_eng[word] = len(word_eng)

        for word in darija_vocab:
            if word not in word_darija:
                word_darija[word] = len(word_darija)

        self.eng_vocab = eng_vocab
        self.darija_vocab = darija_vocab
        self.word_eng = word_eng
        self.word_darija = word_darija

        return word_eng, word_darija

    def get_vocab_sizes(self) -> Tuple[int, int]:
        if self.eng_vocab is None or self.darija_vocab is None:
            raise ValueError("Vocabularies not created yet. Call prepare_vocabularies first.")
        return len(self.word_eng), len(self.word_darija)

    def prepare_sentence(self, seq: str, word_dict: Dict[str, int]) -> torch.Tensor:
        tokens = ['<START>']
        for word in seq.split():
            word = word.lower()
            if word in word_dict:
                tokens.append(word)
            else:
                tokens.append('<UNK>')
        tokens.append('<END>')

        idxs = [word_dict[token] for token in tokens]
        return torch.tensor(idxs)

    def clean_punctuation(self, df: pd.DataFrame) -> pd.DataFrame:

        def clean_text(text: str) -> str:
            cleaned_words = []
            for word in text.split():
                word = word.strip('.,!?()[]{}":;')
                if word:
                    cleaned_words.append(word)
            return ' '.join(cleaned_words)

        cleaned_df = df.copy()
        cleaned_df['eng'] = df['eng'].apply(clean_text)
        cleaned_df['darija'] = df['darija'].apply(clean_text)

        return cleaned_df

    def print_vocab_stats(self) -> None:
        if self.eng_vocab is None or self.darija_vocab is None:
            raise ValueError("Vocabularies not created yet. Call prepare_vocabularies first.")

        print("Vocabulary Statistics:")
        print("-" * 50)
        print(f"English vocabulary size: {len(self.word_eng)}")
        print(f"Darija vocabulary size: {len(self.word_darija)}")
        print("\nExample English words:", list(self.eng_vocab)[:5])
        print("Example Darija words:", list(self.darija_vocab)[:5])

    def word_indexed(self, lang_vocab):
        word_lang = {}

        for word in lang_vocab:
            if word not in word_lang:
                word_lang[word] = len(word_lang)

        return word_lang

    def analyze_sentence_lengths(self, df: pd.DataFrame, fixed_padding_length: int = 7, save_path: str = 'plots'):
        """
        Analyze sentence lengths and calculate padding length. Save plots to specified folder.

        Args:
            df: DataFrame with 'eng' and 'darija' columns
            save_path: Path to folder where plots should be saved
        """
        import os

        # Create plots directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Calculate lengths of all sentences
        eng_lengths = df['eng'].str.split().str.len()
        darija_lengths = df['darija'].str.split().str.len()

        # Calculate statistics
        stats = {
            'eng_mean': eng_lengths.mean(),
            'eng_median': eng_lengths.median(),
            'eng_max': eng_lengths.max(),
            'eng_std': eng_lengths.std(),
            'darija_mean': darija_lengths.mean(),
            'darija_median': darija_lengths.median(),
            'darija_max': darija_lengths.max(),
            'darija_std': darija_lengths.std()
        }

        # Calculate recommended padding length
        padding_length = fixed_padding_length

        print("\nSentence Length Statistics:")
        print("-" * 50)
        print(f"English - Mean: {stats['eng_mean']:.1f}, Median: {stats['eng_median']}, Max: {stats['eng_max']}")
        print(
            f"Darija  - Mean: {stats['darija_mean']:.1f}, Median: {stats['darija_median']}, Max: {stats['darija_max']}")
        print(f"\nUsing fixed padding length: {padding_length}")

        # Create separate plots for each language
        # English plot
        plt.figure(figsize=(10, 6))
        plt.hist(eng_lengths, bins=30, alpha=0.7, label='English')
        plt.axvline(padding_length, color='r', linestyle='--', label='Padding Length')
        plt.title('English Sentence Lengths Distribution')
        plt.xlabel('Sentence Length')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'english_lengths_distribution.png'))
        plt.close()

        # Darija plot
        plt.figure(figsize=(10, 6))
        plt.hist(darija_lengths, bins=30, alpha=0.7, label='Darija')
        plt.axvline(padding_length, color='r', linestyle='--', label='Padding Length')
        plt.title('Darija Sentence Lengths Distribution')
        plt.xlabel('Sentence Length')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'darija_lengths_distribution.png'))
        plt.close()

        # Combined plot
        plt.figure(figsize=(12, 6))
        plt.hist(eng_lengths, bins=30, alpha=0.5, label='English', color='blue')
        plt.hist(darija_lengths, bins=30, alpha=0.5, label='Darija', color='orange')
        plt.axvline(padding_length, color='r', linestyle='--', label='Padding Length')
        plt.title('Sentence Lengths Distribution Comparison')
        plt.xlabel('Sentence Length')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'combined_lengths_distribution.png'))
        plt.close()

        # Save statistics to text file
        with open(os.path.join(save_path, 'length_statistics.txt'), 'w') as f:
            f.write("Sentence Length Statistics\n")
            f.write("-" * 50 + "\n")
            f.write(f"English:\n")
            f.write(f"  Mean: {stats['eng_mean']:.1f}\n")
            f.write(f"  Median: {stats['eng_median']}\n")
            f.write(f"  Max: {stats['eng_max']}\n")
            f.write(f"  Standard Deviation: {stats['eng_std']:.1f}\n")
            f.write(f"Darija:\n")
            f.write(f"  Mean: {stats['darija_mean']:.1f}\n")
            f.write(f"  Median: {stats['darija_median']}\n")
            f.write(f"  Max: {stats['darija_max']}\n")
            f.write(f"  Standard Deviation: {stats['darija_std']:.1f}\n")
            f.write(f"Fixed padding length: {padding_length}\n")

        return padding_length, stats

    def prepare_data_splits(self, df, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(df["eng"], df["darija"], test_size=test_size, random_state=4)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=4)

        print("Data Split Sizes:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Testing samples: {len(X_test)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # Add this to PrepareData class
    def print_length_stats(self, df):
        eng_lengths = df['eng'].str.split().str.len()
        darija_lengths = df['darija'].str.split().str.len()
        print(f"English lengths - Mean: {eng_lengths.mean():.2f}, Max: {eng_lengths.max()}")
        print(f"Darija lengths - Mean: {darija_lengths.mean():.2f}, Max: {darija_lengths.max()}")


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, word_eng, word_darija, max_len=15):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.word_eng = word_eng
        self.word_darija = word_darija
        self.max_len = max_len
        self.device = torch.device("cuda")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get sentences
        eng_sentence = str(self.X.iloc[idx])
        darija_sentence = str(self.y.iloc[idx])

        # Tokenize
        eng_tokens = eng_sentence.split()
        darija_tokens = darija_sentence.split()

        # Convert to indices
        eng_indices = [self.word_eng.get(word.lower(), self.word_eng['<UNK>']) for word in eng_tokens]
        darija_indices = [self.word_darija.get(word.lower(), self.word_darija['<UNK>']) for word in darija_tokens]

        # Add special tokens
        eng_indices = [self.word_eng['<START>']] + eng_indices + [self.word_eng['<END>']]
        darija_indices = [self.word_darija['<START>']] + darija_indices + [self.word_darija['<END>']]

        # Pad sequences
        if len(eng_indices) > self.max_len:
            eng_indices = eng_indices[:self.max_len]
        else:
            eng_indices += [self.word_eng['<PAD>']] * (self.max_len - len(eng_indices))

        if len(darija_indices) > self.max_len:
            darija_indices = darija_indices[:self.max_len]
        else:
            darija_indices += [self.word_darija['<PAD>']] * (self.max_len - len(darija_indices))

        # Move tensors to GPU immediately
        return (torch.tensor(eng_indices, dtype=torch.long).to(self.device),
                torch.tensor(darija_indices, dtype=torch.long).to(self.device))