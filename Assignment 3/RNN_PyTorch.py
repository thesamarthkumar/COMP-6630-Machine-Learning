""" 
A second Implementation of the Seq2Seq model in Pytorch
Increases BLEU Score and Token Accuracy by nearly 10%.

We still need to add the loss plots and hyperparameter tuning to observe the effects
on accuracy.
"""
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import evaluate
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Running locally with my NVIDIA GPU.

"""
Dataset Class definition and Preprocessing.
"""
class TextSummaryDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs.to(DEVICE), targets.to(DEVICE)

# This is the main function to handle our data.
def load_and_prepare_data(input_len, summary_len):
    dataset = load_dataset("billsum", split="ca_test")
    train_test = dataset.train_test_split(test_size=0.2)
    train_data, test_data = train_test['train'], train_test['test']

    train_texts = ["<start> " + t[:input_len] + " <end>" for t in train_data['text']]
    train_summaries = ["<start> " + s[:summary_len] + " <end>" for s in train_data['summary']]
    test_texts = ["<start> " + t[:input_len] + " <end>" for t in test_data['text']]
    test_summaries = ["<start> " + s[:summary_len] + " <end>" for s in test_data['summary']]

    input_counter = Counter(" ".join(train_texts + test_texts).split())
    target_counter = Counter(" ".join(train_summaries + test_summaries).split())

    def build_vocab(counter):
        vocab = {'<pad>': 0, '<unk>': 1}
        for word in counter:
            vocab[word] = len(vocab)
        return vocab

    input_vocab = build_vocab(input_counter)
    summary_vocab = build_vocab(target_counter)

    def tokenize(texts, vocab):
        sequences = []
        for line in texts:
            token_ids = [vocab.get(token, vocab['<unk>']) for token in line.split()]
            sequences.append(torch.tensor(token_ids, dtype=torch.long))
        return sequences

    train_input_seq = tokenize(train_texts, input_vocab)
    train_summary_seq = tokenize(train_summaries, summary_vocab)
    test_input_seq = tokenize(test_texts, input_vocab)
    test_summary_seq = tokenize(test_summaries, summary_vocab)

    return train_input_seq, train_summary_seq, test_input_seq, test_summary_seq, input_vocab, summary_vocab

"""
Define the RNN Seq2Seq model with LSTM layers.
"""
class Seq2SeqModel(nn.Module):
    # Initialization
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=256, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.encoder_embed = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.decoder_embed = nn.Embedding(output_vocab_size, embed_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # Forward pass.
    def forward(self, src, tgt):
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.fc.out_features).to(DEVICE)

        encoder_out, (h, c) = self.encoder_lstm(self.dropout(self.encoder_embed(src)))
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            embedded = self.dropout(self.decoder_embed(input_token).unsqueeze(1))
            output, (h, c) = self.decoder_lstm(embedded, (h, c))
            pred = self.fc(output.squeeze(1))
            outputs[:, t] = pred
            input_token = tgt[:, t]  # teacher forcing

        return outputs

# Using a beam search to decode the tokens.
def decode(model, src, vocab, max_len=128, beam_width=3):
    model.eval()
    with torch.no_grad():
        encoder_out, (h, c) = model.encoder_lstm(model.encoder_embed(src))
        start_id = vocab['<start>']
        end_id = vocab.get('<end>', 0)

        beams = [([start_id], 0.0, h, c)]

        for _ in range(max_len):
            new_beams = []
            for seq, score, h, c in beams:
                if seq[-1] == end_id:
                    new_beams.append((seq, score, h, c))
                    continue
                input_token = torch.tensor([[seq[-1]]], device=DEVICE)
                embedded = model.decoder_embed(input_token)
                output, (h_new, c_new) = model.decoder_lstm(embedded, (h, c))
                logits = model.fc(output.squeeze(1))
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_indices = log_probs.topk(beam_width)
                for i in range(beam_width):
                    new_seq = seq + [topk_indices[0, i].item()]
                    new_score = score + topk_log_probs[0, i].item()
                    new_beams.append((new_seq, new_score, h_new, c_new))


            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        return beams[0][0]

# Returning the BLEU score and accuracy.
def calculate_bleu_and_accuracy(model, test_input, test_target, vocab, max_len=128):
    model.eval()
    rev_vocab = {i: w for w, i in vocab.items()}
    bleu = evaluate.load("bleu")
    preds, refs = [], []
    total_tokens, correct_tokens = 0, 0

    for i in range(min(len(test_input), 50)):
        src = test_input[i].unsqueeze(0).to(DEVICE)
        tgt = test_target[i]
        pred_ids = decode(model, src, vocab, max_len=max_len)
        tgt_ids = [idx.item() for idx in tgt if idx.item() not in (0, 1)]
        pred_ids = [idx for idx in pred_ids if idx not in (0, 1)]

        total_tokens += min(len(tgt_ids), len(pred_ids))
        correct_tokens += sum([p == t for p, t in zip(pred_ids, tgt_ids)])

        pred_words = [rev_vocab.get(idx, '') for idx in pred_ids]
        true_words = [rev_vocab.get(idx, '') for idx in tgt_ids]
        preds.append(" ".join(pred_words))
        refs.append([" ".join(true_words)])

    bleu_score = bleu.compute(predictions=preds, references=refs)['bleu']
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return bleu_score, accuracy

# Training sequence for the neural network.
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in train_loader:
            optimizer.zero_grad()
            output = model(src, tgt)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

# Evaluate the model.
def evaluate_model(model, loader, criterion):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for src, tgt in loader:
            output = model(src, tgt)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            loss_total += loss.item()
    model.train()
    return loss_total / len(loader)

# Function to test our model by choosing input and summary sequence lengths. (1024/2048 for input and 128/256 for summary).
# Generally, increasing summary length to 256 hurts performance and takes longer to compute, while increasing input to 2048 doesn't have much effect.
def run():
    random.seed(42)
    input_len, summary_len = 2048, 128
    train_in, train_sum, test_in, test_sum, in_vocab, sum_vocab = load_and_prepare_data(input_len, summary_len)

    train_loader = DataLoader(TextSummaryDataset(train_in, train_sum), batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(TextSummaryDataset(test_in, test_sum), batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = Seq2SeqModel(len(in_vocab), len(sum_vocab)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_model(model, train_loader, val_loader, optimizer, criterion, epochs=25)
    bleu_score, accuracy = calculate_bleu_and_accuracy(model, test_in, test_sum, sum_vocab)
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Token Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    run()
