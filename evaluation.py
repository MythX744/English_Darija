import torch
import os
from models import TranslationModel, LSTM, Peehole, WMC, StackedLSTM
from prepare_data import PrepareData


def translate_sentence(model, sentence, word_eng, word_darija, device, max_len=15):
    model.eval()

    # Prepare input
    tokens = sentence.lower().split()
    indices = [word_eng.get(token, word_eng['<UNK>']) for token in tokens]
    indices = [word_eng['<START>']] + indices + [word_eng['<END>']]
    if len(indices) < max_len:
        indices += [word_eng['<PAD>']] * (max_len - len(indices))
    src_tensor = torch.LongTensor([indices]).to(device)

    translation = []
    with torch.no_grad():
        # Get encoder output
        _, encoder_states = model.encoder(src_tensor)

        # Decode step by step
        decoder_input = torch.tensor([[2]], device=device)  # START token
        decoder_states = encoder_states

        for _ in range(max_len):
            # Get prediction
            prediction, decoder_states = model.decoder(decoder_input, decoder_states)

            # Get best word
            word_idx = prediction.argmax(dim=1).item()

            # Stop if END token
            if word_idx == 3:  # END token
                break

            # PAD, UNK, START, END tokens
            if word_idx > 3:  # Skip PAD, UNK, START, END
                word = list(word_darija.keys())[list(word_darija.values()).index(word_idx)]
                translation.append(word)

            # Next input
            decoder_input = torch.tensor([[word_idx]], device=device)

    return ' '.join(translation)


def evaluate_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Use same dimensions as training
    embedding_dim = 256  # Changed from 64 to match training
    model_configs = {
        'LSTM': {'hidden_size': 256, 'cell_type': LSTM},  # Changed from 64
        'Peehole': {'hidden_size': 256, 'cell_type': Peehole},
        'WMC': {'hidden_size': 256, 'cell_type': WMC},
        'Stacked_LSTM': {
            'hidden_size': 256,
            'cell_type': StackedLSTM,
            'num_layers': 2
        }
    }

    test_cases = [
        ("hello how are you", "salam labas"),
        ("what is your name", "chno smitk"),
        ("i love you", "kanbghik"),
        ("good morning", "sba7 lkhir")
    ]

    for model_name, config in model_configs.items():
        print(f"\nTesting {model_name} model:")
        print("-" * 50)

        # Create model
        model = TranslationModel(
            input_vocab_size=len(word_eng),
            output_vocab_size=len(word_darija),
            embedding_dim=embedding_dim,
            hidden_size=config['hidden_size'],
            cell_type=config['cell_type'],
            num_layers=config.get('num_layers', 1)
        ).to(device)

        # Load trained model
        checkpoint_path = f'checkpoints/{model_name}/best_model.pt'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Test translations
            for eng, expected in test_cases:
                translation = translate_sentence(model, eng, word_eng, word_darija, device)
                print(f"English: {eng}")
                print(f"Expected: {expected}")
                print(f"Got: {translation}\n")
        else:
            print(f"No checkpoint found for {model_name}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    evaluate_models()
