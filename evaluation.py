import torch
import os
import json
from datetime import datetime
from models import TranslationModel, LSTM, Peehole, WMC, StackedLSTM
from prepare_data import PrepareData
from typing import List, Dict, Tuple


def translate_sentence(model, sentence, word_eng, word_darija, device, max_len=15, temperature=0.8):
    model.eval()

    # Prepare input with better unknown word handling
    tokens = ['<START>'] + sentence.lower().split() + ['<END>']
    indices = []
    for token in tokens:
        if token in ['<START>', '<END>']:
            indices.append(word_eng[token])
        else:
            # Handle unknown words better by checking subwords or using UNK
            indices.append(word_eng.get(token, word_eng['<UNK>']))

    # Pad sequence properly
    if len(indices) < max_len:
        indices += [word_eng['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    src_tensor = torch.LongTensor([indices]).to(device)

    translation = []
    with torch.no_grad():
        # Get encoder output and handle states properly
        encoder_outputs, encoder_states = model.encoder(src_tensor)

        if isinstance(model.encoder.lstm, StackedLSTM):
            decoder_states = encoder_states
        else:
            # For other LSTM types, ensure proper state format
            h_n, c_n = encoder_states
            decoder_states = (h_n, c_n)

        # Initialize decoder input
        decoder_input = torch.tensor([[word_darija['<START>']]], device=device)

        for _ in range(max_len):
            # Get prediction with temperature sampling
            prediction, decoder_states = model.decoder(decoder_input, decoder_states)

            # Apply temperature to logits
            prediction = prediction / temperature
            probs = torch.softmax(prediction, dim=1)

            # Sample from the distribution
            word_idx = torch.multinomial(probs, 1).item()

            # Stop if END token
            if word_idx == word_darija['<END>']:
                break

            # Skip special tokens
            if word_idx in [word_darija['<PAD>'], word_darija['<UNK>'], word_darija['<START>']]:
                continue

            # Convert index to word
            word = list(word_darija.keys())[list(word_darija.values()).index(word_idx)]
            translation.append(word)

            # Next input
            decoder_input = torch.tensor([[word_idx]], device=device)

    return ' '.join(translation)


def evaluate_translation(model, test_cases: List[Tuple[str, str]], word_eng: Dict[str, int],
                         word_darija: Dict[str, int], device: torch.device) -> List[Dict]:
    """
    Evaluate translation quality for a set of test cases

    Args:
        model: The translation model
        test_cases: List of (input_text, expected_translation) tuples
        word_eng: English vocabulary dictionary
        word_darija: Darija vocabulary dictionary
        device: torch device

    Returns:
        List of dictionaries containing results for each test case
    """
    results = []

    for eng, expected in test_cases:
        translation = translate_sentence(model, eng, word_eng, word_darija, device)

        # Calculate simple word overlap score
        expected_words = set(expected.split())
        translated_words = set(translation.split())
        overlap = len(expected_words.intersection(translated_words)) / len(expected_words)

        results.append({
            'input': eng,
            'expected': expected,
            'got': translation,
            'word_overlap': f"{overlap:.2%}"
        })

    return results


def load_model(model_name: str, model_config: Dict, word_eng: Dict[str, int],
               word_darija: Dict[str, int], device: torch.device) -> TranslationModel:
    """
    Load a trained model from checkpoint

    Args:
        model_name: Name of the model architecture
        model_config: Configuration dictionary for the model
        word_eng: English vocabulary dictionary
        word_darija: Darija vocabulary dictionary
        device: torch device

    Returns:
        Loaded model
    """
    model = TranslationModel(
        input_vocab_size=len(word_eng),
        output_vocab_size=len(word_darija),
        embedding_dim=model_config['embedding_dim'],
        hidden_size=model_config['hidden_size'],
        cell_type=model_config['cell_type'],
        num_layers=model_config.get('num_layers', 1)
    ).to(device)

    checkpoint_path = f'checkpoints/{model_name}/best_model.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint for {model_name}")
    else:
        print(f"No checkpoint found for {model_name}")

    return model


def evaluate_models(test_cases: List[Tuple[str, str]] = None):
    """Main evaluation function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # Model configurations
    embedding_dim = 256
    model_configs = {
        'LSTM': {
            'embedding_dim': embedding_dim,
            'hidden_size': 256,
            'cell_type': LSTM
        },
        'Peehole': {
            'embedding_dim': embedding_dim,
            'hidden_size': 256,
            'cell_type': Peehole
        },
        'WMC': {
            'embedding_dim': embedding_dim,
            'hidden_size': 256,
            'cell_type': WMC
        },
        'Stacked_LSTM': {
            'embedding_dim': embedding_dim,
            'hidden_size': 256,
            'cell_type': StackedLSTM,
            'num_layers': 2
        }
    }

    # Default test cases if none provided
    if test_cases is None:
        test_cases = [
            ("They're hiding something, I'm sure", "homa mkhbbyin chi haja, ana mti99en"),
            ("what is your name", "chno smitk"),
            ("i love you", "kanbghik"),
            ("good morning", "sba7 lkhir")
        ]

    # Create results directory
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    # Evaluate each model
    all_results = {}
    for model_name, config in model_configs.items():
        print(f"\nEvaluating {model_name} model:")
        print("-" * 50)

        # Load model
        model = load_model(model_name, config, word_eng, word_darija, device)

        # Run evaluation
        results = evaluate_translation(model, test_cases, word_eng, word_darija, device)

        # Print results
        for result in results:
            print(f"Input: {result['input']}")
            print(f"Expected: {result['expected']}")
            print(f"Got: {result['got']}")
            print(f"Word overlap: {result['word_overlap']}\n")

        all_results[model_name] = results

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    evaluate_models()