import torch
from prepare_data import PrepareData, TranslationDataset, create_dataloaders
from models import TranslationModel, WMC, Peehole, LSTM, StackedLSTM
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train import Trainer
import os
import logging
from datetime import datetime


def setup_logging(save_dir):
    """Setup logging configuration"""
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def train_specific_model(model_name, model, train_loader, val_loader, embedding_dim, hidden_size):
    logging.info(f"\nTraining {model_name} model...")

    try:
        # Setup directories
        save_dir = f'checkpoints/{model_name}'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/plots', exist_ok=True)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.001,
            device='cuda'
        )

        # Train model
        train_losses, val_losses = trainer.train(
            num_epochs=10,
            save_dir=save_dir
        )

        # Save final model
        torch.save({
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size
        }, f'{save_dir}/final_model.pt')

        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{model_name} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/plots/{model_name}_training.png')
        plt.close()

        return train_losses, val_losses

    except Exception as e:
        logging.error(f"Error training {model_name} model: {str(e)}")
        return None, None


def main():
    # Create main checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    setup_logging('checkpoints')

    try:
        # Device setup with error handling
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        logging.info(f"Using device: {device}")

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

        # Dataloader setup
        batch_size = 64
        train_loader, val_loader = create_dataloaders(
            X_train, y_train,
            X_val, y_val,
            word_eng, word_darija,
            batch_size=batch_size
        )

        # Model configurations
        model_configs = {
            'LSTM': {'hidden_size': 256, 'cell_type': LSTM},
            'Peehole': {'hidden_size': 256, 'cell_type': Peehole},
            'WMC': {'hidden_size': 256, 'cell_type': WMC},
            'Stacked_LSTM': {'hidden_size': 256, 'cell_type': StackedLSTM, 'num_layers': 2}
        }

        # Train all models
        for model_name, config in model_configs.items():
            logging.info(f"\nStarting training for {model_name} model...")

            try:
                # Create model directory
                os.makedirs(f'checkpoints/{model_name}', exist_ok=True)

                # Initialize model
                model = TranslationModel(
                    input_vocab_size=len(word_eng),
                    output_vocab_size=len(word_darija),
                    embedding_dim=256,
                    hidden_size=256,
                    num_layers=config.get('num_layers', 1),
                    cell_type=config['cell_type']
                ).to(device)

                # Train model
                train_losses, val_losses = train_specific_model(
                    model_name,
                    model,
                    train_loader,
                    val_loader,
                    embedding_dim=256,
                    hidden_size=config['hidden_size']
                )

                if train_losses is not None:
                    # Save training history
                    history = {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'config': {
                            'embedding_dim': 256,
                            'hidden_size': config['hidden_size'],
                            'num_layers': config.get('num_layers', 1)
                        }
                    }
                    torch.save(history, f'checkpoints/{model_name}/training_history.pt')
                    logging.info(f"Successfully completed training {model_name} model")

                # Clean up
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Error in training {model_name} model: {str(e)}")
                continue

        logging.info("Training completed for all models")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")

    def test_translation(model, sentence, word_eng, word_darija, device):
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(
                [word_eng.get(word, word_eng['<UNK>']) for word in sentence.lower().split()]
            ).unsqueeze(0).to(device)

            # Decode output
            decoder_input = torch.tensor([[word_darija['<START>']]], device=device)
            decoded_words = []

            for _ in range(50):  # Max translation length
                output = model.translate(input_tensor, decoder_input)
                next_word_id = output.argmax(dim=2).item()
                if next_word_id == word_darija['<END>']:
                    break
                decoded_words.append(next_word_id)

            return ' '.join([k for k, v in word_darija.items() if v in decoded_words])

    # Example usage after training
    trained_model = TranslationModel(word_eng, word_darija, 256, 256, LSTM, 1)  # Load best model
    sentence = "How are you?"
    translation = test_translation(trained_model, sentence, word_eng, word_darija, device)
    print(f"English: {sentence} -> Darija: {translation}")

if __name__ == '__main__':
    main()