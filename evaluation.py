from datetime import datetime
import json
import torch
import os
from models import TranslationModel, LSTM, Peehole, WMC, StackedLSTM
from prepare_data import PrepareData


def translate_sentence(model, sentence, word_eng, word_darija, device, max_len=40):
    model.eval()

    # Prepare input
    tokens = ['<START>'] + sentence.lower().split() + ['<END>']
    indices = [word_eng.get(token, word_eng['<UNK>']) for token in tokens]
    if len(indices) < max_len:
        indices += [word_eng['<PAD>']] * (max_len - len(indices))
    src_tensor = torch.LongTensor([indices]).to(device)

    # Initialize output
    translation = []
    with torch.no_grad():
        encoder_outputs, encoder_states = model.encoder(src_tensor)
        decoder_input = torch.tensor([[word_darija['<START>']]], device=device)
        decoder_states = encoder_states

        for _ in range(max_len):
            prediction, decoder_states = model.decoder(decoder_input, decoder_states)
            pred_token = prediction.argmax(dim=1).item()

            if pred_token == word_darija['<END>']:
                break

            if pred_token not in [word_darija['<PAD>'], word_darija['<UNK>'],
                                  word_darija['<START>'], word_darija['<END>']]:
                for word, idx in word_darija.items():
                    if idx == pred_token:
                        translation.append(word)
                        break

            decoder_input = torch.tensor([[pred_token]], device=device)

    return ' '.join(translation)


def evaluate_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize results dictionary
    results = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_results": {}
    }

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
    embedding_dim = 64
    model_configs = {
        'LSTM': {'hidden_size': 64, 'cell_type': LSTM},  # Changed from 64
        'Peehole': {'hidden_size': 64, 'cell_type': Peehole},
        'WMC': {'hidden_size': 64, 'cell_type': WMC},
        'Stacked_LSTM': {
            'hidden_size': 64,
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

        # Initialize results for this model
        results["model_results"][model_name] = {
            "config": {
                "embedding_dim": embedding_dim,
                "hidden_size": config['hidden_size'],
                "num_layers": config.get('num_layers', 1)
            },
            "translations": []
        }

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

                # Store results
                results["model_results"][model_name]["translations"].append({
                    "input": eng,
                    "expected": expected,
                    "output": translation
                })

        else:
            print(f"No checkpoint found for {model_name}")

        del model
        torch.cuda.empty_cache()

        # Save results to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'evaluation_results_{timestamp}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    evaluate_models()
