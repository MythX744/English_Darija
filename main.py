import torch
from prepare_data import PrepareData, TranslationDataset
from models import TranslationModel, WMC, Peehole, LSTM, StackedLSTM
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train import Trainer


def train_specific_model(model_name, model, train_loader, val_loader, embedding_dim, hidden_size):
    print(f"\nTraining {model_name} model...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        device='cuda'
    )

    train_losses, val_losses = trainer.train(
        num_epochs=20,
        save_dir=f'checkpoints/{model_name}'
    )

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'checkpoints/{model_name}/plots/{model_name}_training.png')
    plt.close()

    return train_losses, val_losses


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Data preparation
    data_prep = PrepareData()
    df = data_prep.load_and_clean_data(
        dataset="imomayiz/darija-english",
        folder="sentences",
        drop_column="darija_ar",
        output_csv="darija_english.csv"
    )

    df = data_prep.clean_punctuation(df)
    word_eng, word_darija = data_prep.prepare_vocabularies(df)
    data_prep.print_vocab_stats()
    data_prep.print_length_stats(df)

    # Data splits and loaders
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_prep.prepare_data_splits(df)

    batch_size = 32
    train_dataset = TranslationDataset(X_train, y_train, word_eng, word_darija)
    val_dataset = TranslationDataset(X_val, y_val, word_eng, word_darija)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    # Model configurations
    embedding_dim = 64
    model_configs = {
        'LSTM': {'hidden_size': 64, 'cell_type': LSTM},
        'Peehole': {'hidden_size': 64, 'cell_type': Peehole},
        'WMC': {'hidden_size': 64, 'cell_type': WMC},
        'Stacked_LSTM': {'hidden_size': 64, 'cell_type': StackedLSTM, 'num_layers': 2}
    }

    # Train each model
    for model_name, config in model_configs.items():
        print(f"\nTraining {model_name} model...")

        # Create model
        model = TranslationModel(
            input_vocab_size=len(word_eng),
            output_vocab_size=len(word_darija),
            embedding_dim=embedding_dim,
            hidden_size=config['hidden_size'],
            cell_type=config['cell_type']
        ).to(device)

        # Train model
        train_losses, val_losses = train_specific_model(
            model_name,
            model,
            train_loader,
            val_loader,
            embedding_dim,
            config['hidden_size']
        )

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()