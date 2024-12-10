import pandas as pd
import torch
import os
import json
from datetime import datetime
from models import TranslationModel, LSTM, Peehole, WMC, StackedLSTM
from prepare_data import PrepareData
from typing import List, Dict, Tuple
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


def translate_sentence(model, sentence, word_eng, word_darija, device, max_len=15, temperature=0.8):
    model.eval()

    # Prepare input with better unknown word handling
    tokens = ['<START>'] + sentence.lower().split() + ['<END>']
    indices = []
    for token in tokens:
        if token in ['<START>', '<END>']:
            indices.append(word_eng[token])
        else:
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
            h_n, c_n = encoder_states
            decoder_states = (h_n, c_n)

        decoder_input = torch.tensor([[word_darija['<START>']]], device=device)

        for _ in range(max_len):
            prediction, decoder_states = model.decoder(decoder_input, decoder_states)
            prediction = prediction / temperature
            probs = torch.softmax(prediction, dim=1)
            word_idx = torch.multinomial(probs, 1).item()

            if word_idx == word_darija['<END>']:
                break

            if word_idx in [word_darija['<PAD>'], word_darija['<UNK>'], word_darija['<START>']]:
                continue

            word = list(word_darija.keys())[list(word_darija.values()).index(word_idx)]
            translation.append(word)

            decoder_input = torch.tensor([[word_idx]], device=device)

    return ' '.join(translation)


def calculate_metrics(predicted: str, expected: str) -> Dict[str, float]:
    """Calculate multiple evaluation metrics for a translation"""
    # Word overlap
    expected_words = set(expected.split())
    translated_words = set(predicted.split())
    overlap = len(expected_words.intersection(translated_words)) / len(expected_words)

    # BLEU score
    bleu = sentence_bleu([expected.split()], predicted.split())

    # Length ratio (measure of translation length accuracy)
    length_ratio = len(predicted.split()) / len(expected.split())

    return {
        'word_overlap': overlap,
        'bleu_score': bleu,
        'length_ratio': length_ratio
    }


def evaluate_translation(model, test_cases: List[Tuple[str, str]], word_eng: Dict[str, int],
                         word_darija: Dict[str, int], device: torch.device) -> List[Dict]:
    """Evaluate translation quality for a set of test cases"""
    results = []

    for eng, expected in test_cases:
        translation = translate_sentence(model, eng, word_eng, word_darija, device)
        metrics = calculate_metrics(translation, expected)

        results.append({
            'input': eng,
            'expected': expected,
            'got': translation,
            'metrics': {
                'word_overlap': f"{metrics['word_overlap']:.2%}",
                'bleu_score': f"{metrics['bleu_score']:.4f}",
                'length_ratio': f"{metrics['length_ratio']:.2f}"
            }
        })

    return results


def load_model(model_name: str, model_config: Dict, device: torch.device) -> Tuple[TranslationModel, Dict, Dict]:
    """Load a trained model and its vocabularies from checkpoint"""
    data_prep = PrepareData()
    df = pd.read_csv("final_df.csv")
    df = data_prep.clean_punctuation(df)
    word_eng, word_darija = data_prep.prepare_vocabularies(df)

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
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    return model, word_eng, word_darija


def evaluate_models(test_cases: List[Tuple[str, str]] = None):
    """Main evaluation function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model configurations
    embedding_dim = 128
    model_configs = {
        'LSTM': {
            'embedding_dim': embedding_dim,
            'hidden_size': 128,
            'cell_type': LSTM
        },
        'Peehole': {
            'embedding_dim': embedding_dim,
            'hidden_size': 128,
            'cell_type': Peehole
        },
        'WMC': {
            'embedding_dim': embedding_dim,
            'hidden_size': 128,
            'cell_type': WMC
        },
        'Stacked_LSTM': {
            'embedding_dim': embedding_dim,
            'hidden_size': 128,
            'cell_type': StackedLSTM,
            'num_layers': 2
        }
    }

    if test_cases is None:
        test_cases = [
            ("They're hiding something, I'm sure", "homa mkhbbyin chi haja, ana mti99en"),
            ("what is your name", "chno smitk"),
            ("i love you", "kanbghik"),
            ("good morning", "sba7 lkhir")
        ]

    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    model_summaries = {}

    for model_name, config in model_configs.items():
        print(f"\nEvaluating {model_name} model:")
        print("-" * 50)

        try:
            model, word_eng, word_darija = load_model(model_name, config, device)
            results = evaluate_translation(model, test_cases, word_eng, word_darija, device)

            # Calculate average metrics
            avg_bleu = np.mean([float(r['metrics']['bleu_score']) for r in results])
            avg_overlap = np.mean([float(r['metrics']['word_overlap'].strip('%')) / 100 for r in results])

            model_summaries[model_name] = {
                'average_bleu': f"{avg_bleu:.4f}",
                'average_word_overlap': f"{avg_overlap:.2%}",
                'parameter_count': sum(p.numel() for p in model.parameters())
            }

            # Print results
            for result in results:
                print(f"Input: {result['input']}")
                print(f"Expected: {result['expected']}")
                print(f"Got: {result['got']}")
                print("Metrics:")
                for metric, value in result['metrics'].items():
                    print(f"  {metric}: {value}")
                print()

            all_results[model_name] = results

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue
        finally:
            if 'model' in locals():
                del model
                torch.cuda.empty_cache()

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_file = os.path.join(results_dir, f"detailed_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save model summaries
    summary_file = os.path.join(results_dir, f"model_summaries_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(model_summaries, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to {results_file}")
    print(f"Model summaries saved to {summary_file}")


if __name__ == "__main__":
    evaluate_models()