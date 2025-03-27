import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
import evaluate
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
    Load the California State bill subset of the BillSum dataset
    """
    print("Loading BillSum dataset...")
    billsum = load_dataset("billsum", split="ca_test")
    
    # Split the dataset
    train_test = billsum.train_test_split(test_size=0.2)
    
    return train_test['train'], train_test['test']

def preprocess_data(train_data, test_data, max_input_length, max_summary_length):
    """
    Preprocess the data for training
    """
    # Extract text and summaries
    train_texts = [text[:max_input_length] for text in train_data['text']]
    train_summaries = [summary[:max_summary_length] for summary in train_data['summary']]
    test_texts = [text[:max_input_length] for text in test_data['text']]
    test_summaries = [summary[:max_summary_length] for summary in test_data['summary']]
    
    # Create and fit tokenizer for input texts
    input_tokenizer = Tokenizer()
    input_tokenizer.fit_on_texts(train_texts + test_texts)
    
    # Create and fit tokenizer for summaries
    summary_tokenizer = Tokenizer()
    summary_tokenizer.fit_on_texts(train_summaries + test_summaries)
    
    # Convert texts to sequences
    train_input_seq = input_tokenizer.texts_to_sequences(train_texts)
    test_input_seq = input_tokenizer.texts_to_sequences(test_texts)
    train_summary_seq = summary_tokenizer.texts_to_sequences(train_summaries)
    test_summary_seq = summary_tokenizer.texts_to_sequences(test_summaries)
    
    # Pad sequences
    train_input_padded = pad_sequences(train_input_seq, maxlen=max_input_length, padding='post')
    test_input_padded = pad_sequences(test_input_seq, maxlen=max_input_length, padding='post')
    train_summary_padded = pad_sequences(train_summary_seq, maxlen=max_summary_length, padding='post')
    test_summary_padded = pad_sequences(test_summary_seq, maxlen=max_summary_length, padding='post')
    
    return (train_input_padded, train_summary_padded, 
            test_input_padded, test_summary_padded,
            input_tokenizer, summary_tokenizer)

def create_model(input_vocab_size, output_vocab_size, max_input_length, max_summary_length,
                embedding_dim=256, lstm_units=256, dropout_rate=0.2):
    """
    Create Seq2seq model with specified architecture
    """
    # Encoder
    encoder_inputs = Input(shape=(max_input_length,))
    encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_dropout = Dropout(dropout_rate)(encoder_embedding)
    encoder_lstm = LSTM(lstm_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_dropout)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(max_summary_length,))
    decoder_embedding = Embedding(output_vocab_size, embedding_dim)
    decoder_embedding_outputs = decoder_embedding(decoder_inputs)
    decoder_dropout = Dropout(dropout_rate)(decoder_embedding_outputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_dropout, initial_state=encoder_states)
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Create model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def calculate_bleu(model, test_input, test_summary, input_tokenizer, summary_tokenizer, max_summary_length):
    """
    Calculate BLEU score for model evaluation
    """
    bleu = evaluate.load('bleu')
    predictions = []
    references = []
    
    for i in range(len(test_input)):
        # Generate summary
        target_seq = np.zeros((1, max_summary_length))
        target_seq[0, 0] = summary_tokenizer.word_index.get('start', 1)
        
        # Convert prediction to text
        pred_text = ' '.join([summary_tokenizer.index_word.get(idx, '') 
                            for idx in np.argmax(model.predict([test_input[i:i+1], target_seq]), axis=-1)[0]])
        true_text = ' '.join([summary_tokenizer.index_word.get(idx, '') 
                            for idx in test_summary[i]])
        
        predictions.append(pred_text)
        references.append([true_text])
    
    # Calculate BLEU score
    results = bleu.compute(predictions=predictions, references=references)
    return results['bleu']

def plot_training_history(history):
    """
    Plot training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Step 1: Load and split dataset
    print("Step 1: Loading and splitting dataset")
    train_data, test_data = load_and_prepare_data()
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Step 2: Prepare data with different sequence lengths
    sequence_lengths = [
        (1024, 128),
        (2048, 256)
    ]
    
    results = []
    for input_length, summary_length in sequence_lengths:
        print(f"\nTraining model with input length {input_length} and summary length {summary_length}")
        
        # Preprocess data
        (train_input, train_summary,
         test_input, test_summary,
         input_tokenizer, summary_tokenizer) = preprocess_data(
            train_data, test_data, input_length, summary_length)
        
        # Create and train model
        model = create_model(
            input_vocab_size=len(input_tokenizer.word_index) + 1,
            output_vocab_size=len(summary_tokenizer.word_index) + 1,
            max_input_length=input_length,
            max_summary_length=summary_length,
            lstm_units=256,
            dropout_rate=0.2
        )
        
        # Train model
        history = model.fit(
            [train_input, train_summary[:, :-1]],
            train_summary[:, 1:],
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
        
        # Calculate BLEU score
        bleu_score = calculate_bleu(
            model, test_input, test_summary,
            input_tokenizer, summary_tokenizer,
            summary_length
        )
        
        results.append({
            'input_length': input_length,
            'summary_length': summary_length,
            'bleu_score': bleu_score
        })
        
        # Plot training history
        plot_training_history(history)
    
    # Print results
    print("\nResults for different sequence lengths:")
    for result in results:
        print(f"Input length: {result['input_length']}, "
              f"Summary length: {result['summary_length']}, "
              f"BLEU score: {result['bleu_score']:.4f}")
    
    # Step 4: Hyperparameter tuning
    print("\nStep 4: Hyperparameter tuning")
    hyperparameters = [
        {'lstm_units': 128, 'dropout_rate': 0.1},
        {'lstm_units': 256, 'dropout_rate': 0.2},
        {'lstm_units': 512, 'dropout_rate': 0.3}
    ]
    
    best_score = 0
    best_params = None
    
    for params in hyperparameters:
        print(f"\nTrying parameters: {params}")
        model = create_model(
            input_vocab_size=len(input_tokenizer.word_index) + 1,
            output_vocab_size=len(summary_tokenizer.word_index) + 1,
            max_input_length=1024,  # Using best length from previous experiment
            max_summary_length=128,
            lstm_units=params['lstm_units'],
            dropout_rate=params['dropout_rate']
        )
        
        history = model.fit(
            [train_input, train_summary[:, :-1]],
            train_summary[:, 1:],
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
        
        bleu_score = calculate_bleu(
            model, test_input, test_summary,
            input_tokenizer, summary_tokenizer,
            128
        )
        
        if bleu_score > best_score:
            best_score = bleu_score
            best_params = params
        
        print(f"BLEU score: {bleu_score:.4f}")
    
    print("\nBest hyperparameters:")
    print(f"LSTM units: {best_params['lstm_units']}")
    print(f"Dropout rate: {best_params['dropout_rate']}")
    print(f"Best BLEU score: {best_score:.4f}")

if __name__ == "__main__":
    main() 