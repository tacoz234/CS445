#!/usr/bin/env python3
"""
Naive Bayes Classifier Assignment Solutions
CS445 PA2 - Cole Determan & Ben Berry
"""

import csv
import re
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


class NaiveBayesClassifier:
    """
    Simple Naive Bayes classifier with no Laplace smoothing, used as a base.
    """

    def __init__(self):
        self.class_priors = {}
        self.word_probs = {}
        self.vocabulary = set()
        self.classes = []

    def fit(self, X, y):
        """
        Train the classifier.

        Args:
            X (list): List of tokenized documents (list of words)
            y (list): List of class labels
        """
        # Store classes
        self.classes = list(set(y))
        n_samples = len(X)

        # Build vocabulary from all documents
        for doc in X:
            self.vocabulary.update(doc)

        # Calculate class priors
        class_counts = Counter(y)
        for class_label in self.classes:
            self.class_priors[class_label] = class_counts[class_label] / n_samples

        # Class conditional probabilities
        self.word_probs = {class_label: {} for class_label in self.classes}

        for class_label in self.classes:
            # Get all documents for this class
            class_docs = [X[i] for i in range(len(X)) if y[i] == class_label]

            # Count word frequencies in this class
            word_counts = Counter()
            total_words = 0

            for doc in class_docs:
                word_counts.update(doc)
                total_words += len(doc)

            # Calculate probabilities
            for word in self.vocabulary:
                word_count = word_counts[word]
                if total_words > 0:
                    self.word_probs[class_label][word] = word_count / total_words
                else:
                    self.word_probs[class_label][word] = 0

    def predict_single(self, document):
        """
        Predict the class for a single document.

        Args:
            document (list): Tokenized document (list of words)

        Returns:
            str: Predicted class label
        """
        class_scores = {}

        # Only use words that have non-zero probability in ALL classes
        usable_words = []
        for word in document:
            if word in self.vocabulary:
                has_nonzero_in_all_classes = True
                for class_label in self.classes:
                    if self.word_probs[class_label][word] == 0:
                        has_nonzero_in_all_classes = False
                        break
                if has_nonzero_in_all_classes:
                    usable_words.append(word)

        # Calculate scores for each class
        for class_label in self.classes:
            # Start with log of class prior
            log_prob = math.log(self.class_priors[class_label])

            # Add log probabilities of usable words only
            for word in usable_words:
                word_prob = self.word_probs[class_label][word]
                log_prob += math.log(word_prob)

            class_scores[class_label] = log_prob

        # Return class with highest score
        return max(class_scores, key=class_scores.get)

    def predict(self, X):
        """
        Predict classes for multiple documents.

        Args:
            X (list): List of tokenized documents

        Returns:
            list: List of predicted class labels
        """
        return [self.predict_single(doc) for doc in X]


# Get script directory for file loading (assuming data files are co-located)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()


class ExtendedNaiveBayesClassifier(NaiveBayesClassifier):
    """
    Extended Naive Bayes classifier with Laplace smoothing and additional
    methods for assignment problems.
    
    Overrides fit and introduces a new score-based prediction with smoothing.
    """
    
    def __init__(self):
        super().__init__()
        self.word_counts = {}  # Stores raw counts for smoothing denominator
    
    def fit(self, X, y):
        """
        Override fit to perform standard training and also store raw word counts
        for use in Laplace smoothing during prediction.
        """
        # Run base fit to get priors, vocabulary, and class list
        super().fit(X, y) 
        
        # Recalculate word probabilities using Laplace smoothing (alpha=1)
        self.word_counts = {class_label: Counter() for class_label in self.classes}
        
        for class_label in self.classes:
            class_docs = [X[i] for i in range(len(X)) if y[i] == class_label]
            
            word_counts_class = Counter()
            total_words_class = 0
            for doc in class_docs:
                word_counts_class.update(doc)
                total_words_class += len(doc)
            
            self.word_counts[class_label] = word_counts_class
            
            # Recalculate word_probs with Laplace smoothing
            V = len(self.vocabulary)
            
            # Store smoothed probabilities for words encountered during training
            self.word_probs[class_label] = {}
            for word in self.vocabulary:
                numerator = word_counts_class[word] + 1  # Count + alpha (alpha=1)
                denominator = total_words_class + V      # Total words + V
                self.word_probs[class_label][word] = numerator / denominator
    
    def predict_single_with_scores(self, document):
        """
        Predict a single document and return both prediction and raw scores,
        using Laplace smoothing for all word probabilities.
        
        Args:
            document: List of tokens
            
        Returns:
            tuple: (predicted_class, scores_dict)
        """
        if not self.class_priors:
            raise ValueError("Classifier must be trained before making predictions")
        
        scores = {}
        V = len(self.vocabulary)
        
        for class_label in self.class_priors:
            score = math.log(self.class_priors[class_label])
            
            # Total word count in this class (needed for denominator for OOV words)
            total_words_class = sum(self.word_counts[class_label].values())

            for word in document:
                if word in self.word_probs[class_label]:
                    # Word seen in training: use stored smoothed probability
                    prob = self.word_probs[class_label][word]
                else:
                    # Word NOT seen in training (Out of Vocabulary - OOV)
                    # Count is 0, so numerator is 0 + 1 (alpha)
                    # Denominator is total_words + V (vocabulary size)
                    # Note: V includes all words seen in *all* training data.
                    prob = 1.0 / (total_words_class + V)
            
                # Add log probability to the score
                score += math.log(prob)
            
            scores[class_label] = score
        
        predicted_class = max(scores, key=scores.get)
        
        return predicted_class, scores
    
    def predict_with_scores(self, X):
        """
        Predict multiple documents and return both predictions and scores.
        
        Args:
            X: List of documents (each document is a list of tokens)
            
        Returns:
            tuple: (predictions_list, scores_list)
        """
        predictions = []
        scores_list = []
        
        for document in X:
            pred, scores = self.predict_single_with_scores(document)
            predictions.append(pred)
            scores_list.append(scores)
        
        return predictions, scores_list
    
    def get_probability_distribution(self, document):
        """
        Get probability distribution over classes for a document using the
        Softmax function on the raw log-scores.
        
        Args:
            document: List of tokens
            
        Returns:
            dict: Probability distribution over classes
        """
        _, scores = self.predict_single_with_scores(document)
        
        score_values = np.array(list(scores.values()))
        class_labels = list(scores.keys())
        
        # Softmax: exp(score_i - max_score) / sum(exp(score_j - max_score))
        # Subtracting max_score ensures numerical stability by preventing overflow
        # from large exponents.
        exp_scores = np.exp(score_values - np.max(score_values))
        probabilities_array = exp_scores / np.sum(exp_scores)
        
        probabilities = {class_labels[i]: probabilities_array[i] for i in range(len(class_labels))}
        return probabilities
    
    def get_most_predictive_words(self, n_words=10):
        """
        Find the most predictive words for each class.
        Score = log(P(word|class)) - log(Avg(P(word|other_classes))).
        
        Args:
            n_words: Number of top words to return for each class
            
        Returns:
            dict: Dictionary with class labels as keys and lists of (word, score) tuples as values
        """
        if not self.word_probs:
            raise ValueError("Classifier must be trained before finding predictive words")
        
        predictive_words = {}
        
        for class_label in self.class_priors:
            word_scores = []
            
            for word in self.vocabulary:
                class_prob = self.word_probs[class_label].get(word, 0) 
                
                # Ensure the word has a probability > 0 in the target class
                if class_prob > 0:
                    other_probs = []
                    for other_class in self.class_priors:
                        if other_class != class_label:
                            other_prob = self.word_probs[other_class].get(word, 0)
                            if other_prob > 0:
                                other_probs.append(other_prob)
                    
                    if other_probs:
                        # Calculate the average probability of the word in other classes
                        avg_other_prob = np.mean(other_probs)
                        # Score is the log-ratio of P(word|target) to P(word|others)
                        score = np.log(class_prob) - np.log(avg_other_prob)
                        word_scores.append((word, score))
            
            word_scores.sort(key=lambda x: x[1], reverse=True)
            predictive_words[class_label] = word_scores[:n_words]
        
        return predictive_words


def tokenize_text(text):
    """
    Simple tokenization function that:
    1. Converts to lowercase
    2. Removes punctuation and numbers
    3. Splits on whitespace
    4. Filters out very short words

    Args:
        text (str): Input text

    Returns:
        list: List of tokens
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers, keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Split on whitespace and filter out short words
    tokens = [word for word in text.split() if len(word) > 2]

    return tokens


def load_and_process_data(filepath):
    """
    Load and process a csv file containing labeled text. Labels in the first 
    column, text in the second.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        tuple: (tokenized_texts, labels)
    """
    labels = []
    texts = []
    
    # Prepend the script directory to the filename
    full_path = os.path.join(SCRIPT_DIR, filepath)

    # Read the tab-delimited file
    try:
        with open(full_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=",")
            # Skip header row if present, assuming no header for simple data loading
            # For data_rt_train.csv/test.csv, the first row is data, so don't skip
            
            for row in reader:
                if len(row) >= 2:
                    label = row[0].strip()
                    text = row[1].strip()
                    labels.append(label)
                    texts.append(text)
    except FileNotFoundError:
        print(f"Error: Data file not found at {full_path}. Please check file location.")
        return [], []


    # Tokenize all texts
    tokenized_texts = [tokenize_text(text) for text in texts]

    return tokenized_texts, labels

# ----------------------------------------------------------------------
# ASSIGNMENT PROBLEM FUNCTIONS
# ----------------------------------------------------------------------

def problem_1_worst_movie():
    """
    Problem 1: Find the most "rotten" movie review in the test set.
    """
    print("=" * 60)
    print("PROBLEM 1: Finding the Most Rotten Movie Review")
    print("=" * 60)
    
    train_path = "data_rt_train.csv"
    test_path = "data_rt_test.csv"
    
    X_train, y_train = load_and_process_data(train_path)
    X_test, y_test = load_and_process_data(test_path)
    
    if not X_train or not X_test: return

    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    predictions, scores_list = classifier.predict_with_scores(X_test)
    
    min_rotten_score = float('inf')
    worst_review_idx = -1
    
    # '0' is the label for 'Rotten'
    for i, scores in enumerate(scores_list):
        if '0' in scores:
            rotten_score = scores['0']
            if rotten_score < min_rotten_score:
                min_rotten_score = rotten_score
                worst_review_idx = i
    
    if worst_review_idx == -1:
        print("Could not find a rotten score in the test set.")
        return

    # Read the text of the review from the file
    full_test_path = os.path.join(SCRIPT_DIR, test_path)
    try:
        with open(full_test_path, 'r', encoding='utf-8') as file:
            test_lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: Data file not found at {full_test_path}. Cannot retrieve original text.")
        return
        
    # The index needs to be adjusted if a header was present and skipped, 
    # but based on the loading function, it should match the list index.
    if worst_review_idx < len(test_lines):
        worst_review_line = test_lines[worst_review_idx].strip()
        # Splitting only once to preserve commas in the review text
        actual_label, review_text = worst_review_line.split(',', 1) 
    else:
        actual_label = y_test[worst_review_idx]
        review_text = "Original review text not found in file."


    print(f"Most Rotten Review (Index {worst_review_idx}):")
    print(f"Actual Label: {actual_label} ({'Rotten' if actual_label == '0' else 'Fresh'})")
    print(f"Predicted Label: {predictions[worst_review_idx]} ({'Rotten' if predictions[worst_review_idx] == '0' else 'Fresh'})")
    print(f"Review Text: {review_text}")
    print(f"Rotten Score: {scores_list[worst_review_idx]['0']:.4f}")
    print(f"Fresh Score: {scores_list[worst_review_idx]['1']:.4f}")
    print()


def problem_2_jurassic_park():
    """
    Problem 2: Analyze the Jurassic Park review.
    """
    print("=" * 60)
    print("PROBLEM 2: Jurassic Park Review Analysis")
    print("=" * 60)
    
    train_path = "data_rt_train.csv"
    X_train, y_train = load_and_process_data(train_path)
    
    if not X_train: return
    
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    jurassic_review = "Jurassic Park is a cautionary tale about science gone wrong and filmmaking gone lazy. For all its groundbreaking effects, the plot is held together with dino-sized leaps in logic and characters who make decisions so dumb they deserve to be eaten. The kids are annoying, the adults are incompetent, and Jeff Goldblum spends half the movie shirtless and smirking like he's in a cologne ad. It's less a thrilling adventure and more a theme park ride with a script written on the back of a napkin."
    
    tokenized_review = tokenize_text(jurassic_review)
    
    prob_dist = classifier.get_probability_distribution(tokenized_review)
    
    print("Jurassic Park Review Analysis:")
    print(f"Review length: {len(jurassic_review)} characters, {len(tokenized_review)} tokens")
    print(f"\nProbability Distribution:")
    print(f"P(Rotten|review) = {prob_dist.get('0', 0):.4f}")
    print(f"P(Fresh|review) = {prob_dist.get('1', 0):.4f}")
    
    prediction, raw_scores = classifier.predict_single_with_scores(tokenized_review)
    print(f"\nPredicted Class: {prediction} ({'Rotten' if prediction == '0' else 'Fresh'})")
    
    print(f"Raw log scores - Rotten: {raw_scores.get('0', float('nan')):.4f}, Fresh: {raw_scores.get('1', float('nan')):.4f}")
    print()


def problem_3_roc_curve():
    """
    Problem 3: Generate ROC curve.
    """
    print("=" * 60)
    print("PROBLEM 3: ROC Curve Generation")
    print("=" * 60)
    
    train_path = "data_rt_train.csv"
    test_path = "data_rt_test.csv"
    
    X_train, y_train = load_and_process_data(train_path)
    X_test, y_test = load_and_process_data(test_path)
    
    if not X_train or not X_test: return

    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    _, scores_list = classifier.predict_with_scores(X_test)
    
    # We use the raw log score for the positive class ('1' for Fresh) as the decision function score
    fresh_scores = np.array([scores.get('1', 0) for scores in scores_list])
    
    # Convert labels to 0 and 1 (0=Rotten, 1=Fresh) for scikit-learn's roc_curve
    y_test_array = np.array([int(label) for label in y_test])
    
    fpr, tpr, thresholds = roc_curve(y_test_array, fresh_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Naive Bayes Movie Review Classifier')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(SCRIPT_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.show() # Display the plot
    
    print(f"ROC curve generated and saved as '{roc_path}'")
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print()


def problem_4_predictive_words():
    """
    Problem 4: Find most predictive words.
    """
    print("=" * 60)
    print("PROBLEM 4: Most Predictive Words")
    print("=" * 60)
    
    train_path = "data_rt_train.csv"
    X_train, y_train = load_and_process_data(train_path)
    
    if not X_train: return

    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    predictive_words = classifier.get_most_predictive_words(10)
    
    print("10 Most Indicative Words for ROTTEN reviews (class '0'):")
    # Check if '0' is a key before iterating
    if '0' in predictive_words:
        for i, (word, score) in enumerate(predictive_words['0'], 1):
            print(f"{i:2d}. {word:15s} (score: {score:8.4f})")
    
    print("\n10 Most Indicative Words for FRESH reviews (class '1'):")
    # Check if '1' is a key before iterating
    if '1' in predictive_words:
        for i, (word, score) in enumerate(predictive_words['1'], 1):
            print(f"{i:2d}. {word:15s} (score: {score:8.4f})")
    print()


def problem_5_deceptive_review():
    """
    Problem 5: Write a positive review that gets classified as highly negative.
    """
    print("=" * 60)
    print("PROBLEM 5: Deceptive Positive Review")
    print("=" * 60)
    
    train_path = "data_rt_train.csv"
    X_train, y_train = load_and_process_data(train_path)
    
    if not X_train: return

    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    # Get the top 20 rotten words to construct the deceptive review
    predictive_words = classifier.get_most_predictive_words(20)
    rotten_words = [word for word, _ in predictive_words.get('0', [])]
    
    print("Most rotten-indicative words to potentially use:", rotten_words[:15])
    
    # Construct a review using many Rotten-indicative words but with a positive tone/context
    deceptive_review = """This movie is absolutely boring and tedious, but in the most 
    brilliant way possible! The plot is completely silly and stupid, yet somehow 
    manages to be pure genius. Every dull and awful moment is crafted with such 
    terrible precision that it becomes utterly magnificent. The bad acting is 
    so wonderfully horrible that it transcends into art. This mess of a film 
    is beautifully ugly and perfectly imperfect. A truly awful masterpiece 
    that fails so spectacularly it succeeds beyond measure."""
    
    tokenized_review = tokenize_text(deceptive_review)
    prob_dist = classifier.get_probability_distribution(tokenized_review)
    prediction, raw_scores = classifier.predict_single_with_scores(tokenized_review)
    
    print("\nDeceptive Positive Review:")
    print(f'"{deceptive_review.replace("    ", "")}"')
    print(f"\nClassifier Results:")
    print(f"Predicted Class: {prediction} ({'Rotten' if prediction == '0' else 'Fresh'})")
    print(f"Probability Distribution:")
    print(f"  P(Rotten|review) = {prob_dist.get('0', 0):.4f}")
    print(f"  P(Fresh|review) = {prob_dist.get('1', 0):.4f}")
    print(f"Raw log scores - Rotten: {raw_scores.get('0', float('nan')):.4f}, Fresh: {raw_scores.get('1', float('nan')):.4f}")
    print()


def main():
    """
    Main function to run both the simplified test and all assignment problems.
    """
    
    # --- Simplified Classifier Test (Original main function) ---
    print("=" * 60)
    print("SIMPLIFIED NAIVE BAYES CLASSIFIER TEST (NO SMOOTHING)")
    print("=" * 60)

    # Load and process data
    print("Loading and processing data...")
    X_train, y_train = load_and_process_data("data_rt_train.csv")
    X_test, y_test = load_and_process_data("data_rt_test.csv")
    
    if X_train and X_test:
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Training set distribution: {Counter(y_train)}")
        print(f"Test set distribution: {Counter(y_test)}")

        # Train the simplified Naive Bayes classifier
        print("\nTraining Naive Bayes classifier...")
        nb_classifier = NaiveBayesClassifier()
        nb_classifier.fit(X_train, y_train)

        print(f"Vocabulary size: {len(nb_classifier.vocabulary)}")
        print(f"Class priors: {nb_classifier.class_priors}")

        # Make predictions on test set
        print("\nMaking predictions on test set...")
        y_pred = nb_classifier.predict(X_test)

        # Calculate and report results
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    
    # --- Assignment Problems (Extended Classifier with Smoothing) ---
    print("\n" * 2)
    print("=" * 60)
    print("EXTENDED NAIVE BAYES CLASSIFIER (ASSIGNMENT SOLUTIONS)")
    print("=" * 60)
    
    problem_1_worst_movie()
    problem_2_jurassic_park()
    problem_3_roc_curve()
    problem_4_predictive_words()
    problem_5_deceptive_review()


if __name__ == "__main__":
    main()