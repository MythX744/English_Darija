import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random

from models import TranslationModel, LSTM
from prepare_data import PrepareData, TranslationDataset, collate_fn

import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import os
import json


















if __name__ == '__main__':
    data_prep = Prepare()
    df = data_prep.load_and_clean_data(
        dataset="imomayiz/darija-english",
        folder="sentences",
        drop_column="darija_ar",
        output_csv="darija_english_cleaned.csv"
    )

    # Create vocabularies
    word_eng, word_darija = data_prep.prepare_vocabularies(df, min_freq=2)

    # Validate vocabulary and data
    data_prep.validate_vocab_and_data(df)

    # Split dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_prep.prepare_data_splits(df)

    batch_size = 32
    max_length = 15

    # Create datasets
    train_dataset = Dataset(X_train, y_train, word_eng, word_darija)
    val_dataset = Dataset(X_val, y_val, word_eng, word_darija)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn2)

    # Model parameters
    input_vocab_size = len(word_eng)
    output_vocab_size = len(word_darija)
    embedding_dim = 256
    hidden_dim = 512

    # Instantiate model
    model = TranslationLSTM(
        input_vocab_size=len(word_eng),
        output_vocab_size=len(word_darija),
        embedding_dim=256,
        hidden_dim=512
    ).cuda()

    # Train with improved settings
    trained_model = train_model(model, train_loader, val_loader, "cuda", num_epochs=20)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=20)

    test_sentences = [
        "Hello, how are you?",
        "What is your name?",
        "Thank you very much",
        "Good morning"
    ]

    for sentence in test_sentences:
        translation = translate_sentence(sentence, trained_model, word_eng, word_darija, device)
        print(f"English: {sentence}")
        print(f"Darija: {translation}\n")

