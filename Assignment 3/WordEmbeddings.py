import numpy as np
import gensim.downloader as api
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def load_glove_model():
    """
    Load the Glove-twitter-50D model
    """
    print("Loading Glove-twitter-50D model...")
    return api.load('glove-twitter-50')

def create_similarity_matrix_glove(model, words):
    """
    Create similarity matrix using Glove model
    """
    # Get word vectors
    word_vectors = []
    for word in words:
        try:
            word_vectors.append(model[word.lower()])
        except KeyError:
            print(f"Warning: '{word}' not found in Glove vocabulary")
            word_vectors.append(np.zeros(50))  # Use zero vector for unknown words
    
    # Convert to numpy array
    word_vectors = np.array(word_vectors)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(word_vectors)
    
    return similarity_matrix

def train_fasttext_model(words):
    """
    Train FastText model with the given words
    """
    # Create sentences with context (simple example sentences)
    sentences = [
        ['the', 'dog', 'bark', 'loudly'],
        ['a', 'tree', 'grows', 'tall'],
        ['the', 'bank', 'near', 'river'],
        ['save', 'money', 'in', 'bank'],
        ['river', 'flows', 'by', 'tree'],
        ['dog', 'plays', 'near', 'tree'],
    ]
    
    # Initialize and train FastText model
    model = FastText(vector_size=50, window=5, min_count=1, sentences=sentences, epochs=10)
    return model

def create_similarity_matrix_fasttext(model, words):
    """
    Create similarity matrix using FastText model
    """
    # Get word vectors
    word_vectors = []
    for word in words:
        word_vectors.append(model.wv[word.lower()])
    
    # Convert to numpy array
    word_vectors = np.array(word_vectors)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(word_vectors)
    
    return similarity_matrix

def display_similarity_matrix(matrix, words, title):
    """
    Display similarity matrix as a pandas DataFrame
    """
    df = pd.DataFrame(matrix, index=words, columns=words)
    print(f"\n{title}:")
    print(df.round(4))

def main():
    # Define the words
    words = ['Dog', 'Bark', 'Tree', 'Bank', 'River', 'Money']
    
    # Part A: Glove-twitter-50D
    print("\nPart A: Using Glove-twitter-50D word2vec")
    glove_model = load_glove_model()
    glove_similarity = create_similarity_matrix_glove(glove_model, words)
    display_similarity_matrix(glove_similarity, words, "Glove-twitter-50D Cosine Similarities")
    
    # Part B: FastText
    print("\nPart B: Using FastText")
    fasttext_model = train_fasttext_model(words)
    fasttext_similarity = create_similarity_matrix_fasttext(fasttext_model, words)
    display_similarity_matrix(fasttext_similarity, words, "FastText Cosine Similarities")
    
    # Part C: Compare and analyze
    print("\nPart C: Semantic Analysis")
    print("\nComparison of embeddings:")
    print("1. Glove-twitter-50D:")
    print("   - Pre-trained on Twitter data")
    print("   - Captures general word relationships from social media context")
    print("   - Better for common words and social media language")
    
    print("\n2. FastText:")
    print("   - Trained on our custom minimal context")
    print("   - Uses subword information (n-grams)")
    print("   - Better for out-of-vocabulary words")
    print("   - Can handle morphologically rich languages better")
    
    # Analyze specific word relationships
    print("\nSpecific word relationship analysis:")
    print("- Bank-Money relationship:")
    glove_bank_money = glove_similarity[words.index('Bank')][words.index('Money')]
    fasttext_bank_money = fasttext_similarity[words.index('Bank')][words.index('Money')]
    print(f"  Glove: {glove_bank_money:.4f}, FastText: {fasttext_bank_money:.4f}")
    
    print("- Dog-Bark relationship:")
    glove_dog_bark = glove_similarity[words.index('Dog')][words.index('Bark')]
    fasttext_dog_bark = fasttext_similarity[words.index('Dog')][words.index('Bark')]
    print(f"  Glove: {glove_dog_bark:.4f}, FastText: {fasttext_dog_bark:.4f}")

if __name__ == "__main__":
    main() 