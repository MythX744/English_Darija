import torch
from prepare_data import PrepareData, TranslationDataset
from models import TranslationModel, WMC, Peehole, LSTM, StackedLSTM
import re


class ModelEvaluator:
    def __init__(self, checkpoint_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.data_prep = PrepareData()
        df = self.data_prep.load_and_clean_data(
            dataset="imomayiz/darija-english",
            folder="sentences",
            drop_column="darija_ar",
            output_csv="darija_english.csv"
        )
        df = self.data_prep.clean_punctuation(df)
        self.word_eng, self.word_darija = self.data_prep.prepare_vocabularies(df)
        self.models = {}
        self.load_models()

    def load_models(self):
        model_types = {
            'LSTM': LSTM,
            'Peehole': Peehole,
            'WMC': WMC,
            'Stacked_LSTM': StackedLSTM
        }

        for model_name, cell_type in model_types.items():
            try:
                checkpoint = torch.load(f'{self.checkpoint_dir}/{model_name}/final_model.pt')
                model = TranslationModel(
                    input_vocab_size=len(self.word_eng),
                    output_vocab_size=len(self.word_darija),
                    embedding_dim=checkpoint['embedding_dim'],
                    hidden_size=checkpoint['hidden_size'],
                    cell_type=cell_type
                ).to(self.device)

                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                print(f"Loaded {model_name} model successfully")
                self.models[model_name] = model
            except Exception as e:
                print(f"Failed to load {model_name} model: {str(e)}")

    def translate_with_model(self, sentence, model, max_length=50):
        model.eval()
        with torch.no_grad():
            # Prepare input sequence
            eng_tokens = ['<START>'] + sentence.lower().split() + ['<END>']
            eng_indices = [self.word_eng.get(token, self.word_eng['<UNK>']) for token in eng_tokens]
            eng_tensor = torch.tensor([eng_indices]).to(self.device)
            print(f"Input tensor shape: {eng_tensor.shape}")
            print(f"Input tokens: {eng_tokens}")

            # Initialize with START token
            decoder_input = torch.tensor([[self.word_darija['<START>']]]).to(self.device)
            decoded_words = []
            hidden = None

            # Generate sequence
            for i in range(max_length):
                # Create batch
                batch = {
                    'eng': eng_tensor,
                    'eng_lengths': torch.tensor([len(eng_indices)]).to(self.device),
                    'darija': decoder_input
                }

                try:
                    # Forward pass through the encoder
                    encoder_outputs = model.encoder_embedding(batch['eng'])
                    encoder_output, hidden = model.encoder(encoder_outputs)

                    # Forward pass through the decoder
                    decoder_embedded = model.decoder_embedding(batch['darija'])
                    decoder_output, hidden = model.decoder(decoder_embedded, hidden)
                    output = model.fc(decoder_output)

                    print(f"Step {i}, Output shape: {output.shape}")

                    # Get prediction
                    next_token = output[:, -1].argmax(dim=-1, keepdim=True)

                    # Stop if END token
                    if next_token.item() == self.word_darija['<END>']:
                        print("Generated END token")
                        break

                    # Add token to decoded sequence
                    for word, idx in self.word_darija.items():
                        if idx == next_token.item() and word not in ['<START>', '<END>', '<PAD>', '<UNK>']:
                            decoded_words.append(word)
                            print(f"Generated word: {word}")
                            break

                    # Update decoder input
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)

                except Exception as e:
                    print(f"Error during translation: {str(e)}")
                    break

            return ' '.join(decoded_words) if decoded_words else '<no translation>'

    def evaluate_all_models(self, sentence):
        results = {}
        for model_name, model in self.models.items():
            print(f"\n{'=' * 20} Generating translation with {model_name} {'=' * 20}")
            try:
                translation = self.translate_with_model(sentence, model)
                results[model_name] = translation
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
                results[model_name] = '<error>'
        return results


def main():
    evaluator = ModelEvaluator('checkpoints')
    test_sentences = [
        "how are you?",
        "what is your name?",
        "I love you",
        "thank you very much",
        "good morning"
    ]

    print("Translation Results:\n")
    for sentence in test_sentences:
        print(f"English: {sentence}")
        translations = evaluator.evaluate_all_models(sentence)
        for model_name, translation in translations.items():
            print(f"  {model_name}: {translation}")
        print("-" * 50)


if __name__ == "__main__":
    main()

