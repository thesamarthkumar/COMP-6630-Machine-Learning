""" Implementation in PyTorch """
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_vocab(texts, min_freq=2):
    """
    Builds a vocabulary dict with:
      <pad>:0, <unk>:1, <start>:2, <end>:3
    Then enumerates words that appear >= min_freq times.
    """
    counter = Counter()
    for line in texts:
        counter.update(line.lower().split())

    vocab = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
    return vocab

def tokenize(texts, vocab, max_len):
    """
    Convert each line of text into a list of token IDs.
    Truncate/pad to max_len. Insert <start> and <end>.
    """
    sequences = []
    for line in texts:
        tokens = line.lower().split()
        # Subtract 2 for <start>, <end> inside max_len
        truncated = tokens[: max_len - 2]

        token_ids = [vocab['<start>']]
        for t in truncated:
            token_ids.append(vocab.get(t, vocab['<unk>']))
        token_ids.append(vocab['<end>'])

        sequences.append(torch.tensor(token_ids, dtype=torch.long))
    return sequences

class TextSummaryDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def collate_fn(batch):
    """
    Pad each batch to the max sequence length in that batch.
    """
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs.to(DEVICE), targets.to(DEVICE)
def load_data(input_len, summary_len):
    """
    1. Load the CA subset of BillSum.
    2. Split into train/test with HuggingFace's train_test_split.
    3. Build vocab.
    4. Tokenize text & summary, returning everything.
    """
    # Load "ca_test" portion of BillSum
    dataset = load_dataset("billsum", split="ca_test")
    train_test = dataset.train_test_split(test_size=0.2)
    train_data = train_test["train"]
    test_data = train_test["test"]

    # Truncate text/summary
    train_texts = [text[:input_len] for text in train_data["text"]]
    train_summaries = [summary[:summary_len] for summary in train_data["summary"]]
    test_texts = [text[:input_len] for text in test_data["text"]]
    test_summaries = [summary[:summary_len] for summary in test_data["summary"]]

    # Build vocab from combined sets
    input_vocab = build_vocab(train_texts + test_texts, min_freq=2)
    summary_vocab = build_vocab(train_summaries + test_summaries, min_freq=2)

    # Tokenize
    train_input_seq = tokenize(train_texts, input_vocab, input_len)
    train_summary_seq = tokenize(train_summaries, summary_vocab, summary_len)
    test_input_seq = tokenize(test_texts, input_vocab, input_len)
    test_summary_seq = tokenize(test_summaries, summary_vocab, summary_len)

    return train_input_seq, train_summary_seq, test_input_seq, test_summary_seq, input_vocab, summary_vocab

""" Define the RNN model. """
class Seq2SeqModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size,
                 embedding_dim=256, hidden_dim=512, dropout=0.2):
        super().__init__()
        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src shape: (batch, src_len)
        tgt shape: (batch, tgt_len)
        """
        batch_size, tgt_len = tgt.size()
        vocab_size = self.fc.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=DEVICE)

        # Encode
        embedded_src = self.dropout(self.encoder_embedding(src))
        _, (hidden, cell) = self.encoder_lstm(embedded_src)

        # Decoder first input <start> tokens from tgt
        input_token = tgt[:, 0]  # shape: (batch,)

        for t in range(1, tgt_len):
            # Embed the previous token
            embedded_tgt = self.dropout(self.decoder_embedding(input_token).unsqueeze(1))
            # Pass through LSTM
            dec_output, (hidden, cell) = self.decoder_lstm(embedded_tgt, (hidden, cell))
            # Generate vocab distribution
            predictions = self.fc(dec_output.squeeze(1))
            outputs[:, t] = predictions

            # Decide next input
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = predictions.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs
      
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=100):
    model.train()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for src, tgt in train_loader:
            optimizer.zero_grad()
            output = model(src, tgt, teacher_forcing_ratio=0.5)

            # Reshape for cross-entropy
            # output = (batch, tgt_len, vocab_size)
            # We ignore the first token’s output, so we shift by 1
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion)
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses
  
def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in val_loader:
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            total_loss += loss.item()
    model.train()
    return total_loss / len(val_loader)

def greedy_decode(model, src, summary_vocab, max_len=128):
    """
    Simple greedy decoding (no beam search) for demonstration.
    """
    model.eval()
    with torch.no_grad():
        # Encode
        embedded_src = model.encoder_embedding(src)
        _, (hidden, cell) = model.encoder_lstm(embedded_src)

        # Start token
        start_id = summary_vocab['<start>']
        end_id   = summary_vocab['<end>']

        # Decoder input initially = <start>
        input_token = torch.tensor([start_id], device=DEVICE).unsqueeze(0)  # shape: (1, 1)

        decoded_tokens = []
        for _ in range(max_len):
            embedded_tgt = model.decoder_embedding(input_token)
            dec_output, (hidden, cell) = model.decoder_lstm(embedded_tgt, (hidden, cell))
            preds = model.fc(dec_output.squeeze(1))
            next_token = preds.argmax(1)

            # Accumulate token
            decoded_tokens.append(next_token.item())

            # If <end>, stop
            if next_token.item() == end_id:
                break

            input_token = next_token.unsqueeze(0)  # shape: (1, 1)

    return decoded_tokens

def calculate_bleu(model, test_input, test_target, summary_vocab, limit=50):
    """
    Compare predicted summaries vs. ground truth with BLEU.
    """
    bleu = evaluate.load("bleu")
    rev_summary_vocab = {v: k for k, v in summary_vocab.items()}

    predictions, references = [], []
    for i in range(min(limit, len(test_input))):
        src_seq = test_input[i].unsqueeze(0).to(DEVICE)   # shape: (1, seq_len)
        tgt_seq = test_target[i]                          # shape: (seq_len,)

        # Decode
        pred_ids = greedy_decode(model, src_seq, summary_vocab)
        # Convert IDs -> words (filter out <pad>, <unk>, <start>, <end>)
        pred_tokens = [
            rev_summary_vocab.get(idx, '<unk>') 
            for idx in pred_ids 
            if idx not in (0,1,2,3)
        ]
        # Ground truth tokens
        tgt_tokens = [
            rev_summary_vocab.get(idx.item(), '<unk>') 
            for idx in tgt_seq 
            if idx.item() not in (0,1,2,3)
        ]
        predictions.append(" ".join(pred_tokens))
        references.append([" ".join(tgt_tokens)])  # references is a list of lists

    results = bleu.compute(predictions=predictions, references=references)
    return results["bleu"]

def run_experiment(input_len, summary_len, epochs, batch_size, learning_rate ):
    print(f"\n--- Running experiment: input_len={input_len}, summary_len={summary_len} ---")

    # Load & tokenize
    (train_in, train_sum,
     test_in, test_sum,
     in_vocab, sum_vocab) = load_data(input_len, summary_len)

    # Build datasets/dataloaders
    train_dataset = TextSummaryDataset(train_in, train_sum)
    test_dataset  = TextSummaryDataset(test_in, test_sum)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # Create model
    model = Seq2SeqModel(
        input_vocab_size=len(in_vocab),
        output_vocab_size=len(sum_vocab),
        embedding_dim=256,
        hidden_dim=512,
        dropout=0.2,
        lr = learning_rate
    ).to(DEVICE)

    # Loss + Optim
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Train
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, criterion, epochs=epochs
    )

    # Plot (optional)
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title(f"Loss Curve (input={input_len}, summary={summary_len})")
    plt.show()

    # Evaluate BLEU
    bleu_score = calculate_bleu(model, test_in, test_sum, sum_vocab)
    print(f"BLEU score: {bleu_score:.4f}")
    return bleu_score

""" Training the model for different configurations, evaluate BLEU score. """
results = {}
bleu = run_experiment(1024, 128, epochs=50, batch_size=32)
results[(1024, 128)] = bleu
bleu = run_experiment(2048, 128, epochs=50, batch_size=32)
results[(2048, 128)] = bleu
bleu = run_experiment(1024, 256, epochs=50, batch_size=32)
results[(1024, 256)] = bleu
bleu = run_experiment(2048, 256, epochs=50, batch_size=32)
results[(2048, 256)] = bleu
print("\n=== Final Results ===")
for (i_len, s_len), score in results.items():
    print(f"Input={i_len}, Summary={s_len} => BLEU={score:.4f}")
