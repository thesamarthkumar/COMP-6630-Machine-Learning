''' Works in Python 3.11, but not above.'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

# Load dataset
dataset = load_dataset("billsum", split="ca_test")
texts = dataset["text"]
summaries = dataset["summary"]

# Add <start> and <end> tokens
summaries = ["<start> " + s + " <end>" for s in summaries]

# Split
train_texts, test_texts, train_summaries, test_summaries = train_test_split(
    texts, summaries, test_size=0.2, random_state=42
)

# Tokenization function
def tokenize_texts(texts, max_len):
    tokenizer = Tokenizer(oov_token="<OOV>", filters='')  # Don't strip special tokens
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences, maxlen=max_len, padding="post")
    return tokenizer, sequences

# Sequence lengths
INPUT_SEQ_LEN = 1024
SUMMARY_SEQ_LEN = 256

# Tokenize
text_tokenizer, text_sequences = tokenize_texts(train_texts, INPUT_SEQ_LEN)
summary_tokenizer, summary_sequences = tokenize_texts(train_summaries, SUMMARY_SEQ_LEN)
_, test_text_sequences = tokenize_texts(test_texts, INPUT_SEQ_LEN)

VOCAB_SIZE_TEXT = len(text_tokenizer.word_index) + 1
VOCAB_SIZE_SUMMARY = len(summary_tokenizer.word_index) + 1

# Model parameters
HIDDEN_UNITS = 256
DROPOUT = 0.3

# Encoder
encoder_inputs = Input(shape=(INPUT_SEQ_LEN,))
encoder_embedding = Embedding(VOCAB_SIZE_TEXT, 256, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(HIDDEN_UNITS, return_state=True, dropout=DROPOUT)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Decoder
decoder_inputs = Input(shape=(SUMMARY_SEQ_LEN,))
decoder_embedding = Embedding(VOCAB_SIZE_SUMMARY, 256, mask_zero=True, name="decoder_embedding")(decoder_inputs)
decoder_lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True, dropout=DROPOUT, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(VOCAB_SIZE_SUMMARY, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# Full training model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Prepare decoder target sequences (shifted)
decoder_output_sequences = np.zeros_like(summary_sequences)
decoder_output_sequences[:, :-1] = summary_sequences[:, 1:]

# Train
BATCH_SIZE = 64
EPOCHS = 10

model.fit(
    [text_sequences, summary_sequences],
    decoder_output_sequences,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
)

# Inference encoder model
encoder_model_inf = Model(encoder_inputs, [state_h, state_c])

# Inference decoder model
decoder_state_input_h = Input(shape=(HIDDEN_UNITS,))
decoder_state_input_c = Input(shape=(HIDDEN_UNITS,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb_inf = model.get_layer("decoder_embedding")(decoder_inputs)
decoder_lstm_layer = model.get_layer("decoder_lstm")
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm_layer(
    dec_emb_inf, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_dense_layer = model.get_layer("decoder_dense")
decoder_outputs_inf = decoder_dense_layer(decoder_outputs_inf)

decoder_model_inf = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

# Greedy decoding
def generate_summary(input_text):
    input_seq = text_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=INPUT_SEQ_LEN, padding='post')
    state_values = encoder_model_inf.predict(input_seq)

    target_seq = np.array([[summary_tokenizer.word_index["<start>"]]])
    stop_token = summary_tokenizer.word_index["<end>"]
    decoded_words = []

    for _ in range(SUMMARY_SEQ_LEN):
        output_tokens, h, c = decoder_model_inf.predict([target_seq] + state_values)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = summary_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_token_index == stop_token or sampled_word == '':
            break

        decoded_words.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        state_values = [h, c]

    return ' '.join(decoded_words)

# BLEU evaluation
def evaluate_bleu_fixed(test_texts, test_summaries):
    references = []
    predictions = []

    for i in range(len(test_texts)):
        input_text = test_texts[i]
        reference = test_summaries[i]
        prediction = generate_summary(input_text)

        references.append([reference.replace("<start>", "").replace("<end>", "").lower().split()])
        predictions.append(prediction.lower().split())

        if i < 3:
            print(f"\nExample {i+1}")
            print("Input:", input_text[:300], "...")
            print("Reference:", reference)
            print("Prediction:", prediction)

    return corpus_bleu(references, predictions)

# Run BLEU evaluation
bleu_score = evaluate_bleu_fixed(test_texts, test_summaries)
print(f"\nFinal BLEU Score: {bleu_score:.4f}")

''' OPTIONAL - takes longer.'''
# Print a full summary example
# print("\n=== My Summary Example ===")
# input_text = test_texts[0]
# reference = test_summaries[0]
# generated_summary = generate_summary(input_text)

# print(f"\nOriginal Text (truncated):\n{input_text[:500]}...\n")
# print(f"Reference Summary:\n{reference}")
# print(f"Generated Summary:\n{generated_summary}")
