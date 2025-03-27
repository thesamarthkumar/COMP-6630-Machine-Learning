import nltk
import numpy as np
import pandas as pd
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def download_twitter_data():
    """
    Download Twitter sample data from NLTK
    """
    print("Downloading Twitter sample data...")
    nltk.download('twitter_samples')
    nltk.download('punkt')
    
    # Get positive and negative tweets
    positive_tweets = nltk.corpus.twitter_samples.strings('positive_tweets.json')
    negative_tweets = nltk.corpus.twitter_samples.strings('negative_tweets.json')
    
    # Create labels
    tweets = positive_tweets + negative_tweets
    labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)
    
    return tweets, labels

def split_data(tweets, labels):
    """
    Split data into training (70%) and testing (30%) sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        tweets, labels, 
        test_size=0.3, 
        random_state=42, 
        stratify=labels
    )
    return X_train, X_test, y_train, y_test

def extract_ngrams(text, n):
    """
    Extract n-grams from text
    """
    tokens = word_tokenize(text.lower())
    n_grams = list(ngrams(tokens, n))
    return [' '.join(gram) for gram in n_grams]

def train_and_evaluate_model(X_train, X_test, y_train, y_test, n):
    """
    Train logistic regression model using n-gram features and evaluate performance
    """
    # Create vectorizer for n-grams
    vectorizer = CountVectorizer(ngram_range=(n, n))
    
    # Transform text data to n-gram features
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_features, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_features)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report, len(vectorizer.get_feature_names_out())

def plot_performance(accuracies, feature_counts):
    """
    Plot model performance and feature counts for different n-gram values
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracies
    ax1.plot(range(1, 5), accuracies, marker='o')
    ax1.set_xlabel('n-gram size')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy vs n-gram size')
    ax1.grid(True)
    
    # Plot feature counts
    ax2.bar(range(1, 5), feature_counts)
    ax2.set_xlabel('n-gram size')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Number of Features vs n-gram size')
    
    plt.tight_layout()
    plt.savefig('ngram_analysis.png')
    plt.close()

def main():
    # Download and prepare data
    print("Part (a): Downloading and splitting data")
    tweets, labels = download_twitter_data()
    X_train, X_test, y_train, y_test = split_data(tweets, labels)
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Extract n-grams and evaluate models
    print("\nPart (b) & (c): Extracting n-grams and building models")
    accuracies = []
    feature_counts = []
    
    for n in range(1, 5):
        print(f"\nTraining model with {n}-grams")
        accuracy, report, feature_count = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, n
        )
        accuracies.append(accuracy)
        feature_counts.append(feature_count)
        
        print(f"\n{n}-gram Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print(f"Number of features: {feature_count}")
    
    # Plot and analyze results
    print("\nPart (d): Analyzing n-gram effect on model performance")
    plot_performance(accuracies, feature_counts)
    
    print("\nAnalysis of n-gram effect on model performance:")
    print("1. Accuracy trends:")
    for n, acc in enumerate(accuracies, 1):
        print(f"   {n}-gram: {acc:.4f}")
    
    print("\n2. Feature space size:")
    for n, count in enumerate(feature_counts, 1):
        print(f"   {n}-gram: {count} features")
    
    print("\n3. Key observations:")
    print("   - As n increases, the feature space grows significantly")
    print("   - Higher n-grams capture more context but may lead to sparsity")
    print("   - Trade-off between context capture and computational complexity")
    print("   - Best performing n-gram size: ", np.argmax(accuracies) + 1)
    
    print("\nResults have been saved to 'ngram_analysis.png'")

if __name__ == "__main__":
    main() 