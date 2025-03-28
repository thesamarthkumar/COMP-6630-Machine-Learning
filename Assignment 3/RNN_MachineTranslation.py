import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

# Load dataset from Hugging Face
dataset = load_dataset("billsum", split="ca_test")

# Extract main text and summaries
texts = dataset["text"]
summaries = dataset["summary"]

# Split dataset into training (80%) and testing (20%)
train_texts, test_texts, train_summaries, test_summaries = train_test_split(
    texts, summaries, test_size=0.2, random_state=42
)

# Tokenization
def tokenize_texts(texts, max_len):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences, maxlen=max_len, padding="post")
    return tokenizer, sequences

# Set sequence length
INPUT_SEQ_LEN = 1024  # Change to 1024 or 2048 as per the task
SUMMARY_SEQ_LEN = 256  # Change to 128 or 256

# Tokenize main texts and summaries
text_tokenizer, text_sequences = tokenize_texts(train_texts, INPUT_SEQ_LEN)
summary_tokenizer, summary_sequences = tokenize_texts(train_summaries, SUMMARY_SEQ_LEN)

VOCAB_SIZE_TEXT = len(text_tokenizer.word_index) + 1
VOCAB_SIZE_SUMMARY = len(summary_tokenizer.word_index) + 1

# Build Seq2Seq Model (RNN with LSTM)
HIDDEN_UNITS = 256  # Change as needed
DROPOUT = 0.3  # Experiment with different values

# Encoder
encoder_inputs = Input(shape=(INPUT_SEQ_LEN,))
encoder_embedding = Embedding(VOCAB_SIZE_TEXT, 256, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(HIDDEN_UNITS, return_state=True, dropout=DROPOUT)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Decoder
decoder_inputs = Input(shape=(SUMMARY_SEQ_LEN,))
decoder_embedding = Embedding(VOCAB_SIZE_SUMMARY, 256, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True, dropout=DROPOUT)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

# Output layer
decoder_dense = Dense(VOCAB_SIZE_SUMMARY, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Compile model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Print Model Summary
model.summary()

# Prepare the decoder output (shifted left target sequences)
decoder_output_sequences = np.zeros_like(summary_sequences)
decoder_output_sequences[:, :-1] = summary_sequences[:, 1:]

# Train Model
BATCH_SIZE = 64
EPOCHS = 10  # Change based on experimentation

model.fit(
    [text_sequences, summary_sequences],
    decoder_output_sequences,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
)

# Evaluate BLEU Score

def decode_sequence(sequence, tokenizer):
    """Convert sequence of token IDs to words"""
    words = [tokenizer.index_word.get(idx, "") for idx in sequence if idx > 0]  # Ignore padding
    return " ".join(words)

def evaluate_bleu(model, test_texts, test_summaries, text_tokenizer, summary_tokenizer):
    test_text_sequences = tokenize_texts(test_texts, INPUT_SEQ_LEN)[1]  # Correct input shape for encoder
    test_summary_sequences = tokenize_texts(test_summaries, SUMMARY_SEQ_LEN)[1]  # Correct decoder input shape

    predictions = model.predict([test_text_sequences, test_summary_sequences])  # Fix input mismatch

    # Convert predicted sequences back to words
    predicted_texts = [decode_sequence(np.argmax(pred, axis=1), summary_tokenizer) for pred in predictions]

    # Calculate BLEU score
    bleu_score = corpus_bleu([[ref.split()] for ref in test_summaries], [pred.split() for pred in predicted_texts])
    return bleu_score

# Run BLEU evaluation
bleu_score = evaluate_bleu(model, test_texts, test_summaries, text_tokenizer, summary_tokenizer)
print(f"BLEU Score: {bleu_score}")

bleu_score = evaluate_bleu(model, test_texts, test_summaries, text_tokenizer, summary_tokenizer)
print(f"BLEU Score: {bleu_score}")

# BLEU SCORE IS EXTREMELY LOW - Must Fix. Aside from that, program runs as intended.
