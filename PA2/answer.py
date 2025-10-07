"""
Naive Bayes Classifier Assignment Solutions
CS445 PA2 - Cole Determan
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import roc_curve, auc
from naive_bayes import NaiveBayesClassifier, tokenize_text, load_and_process_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class ExtendedNaiveBayesClassifier(NaiveBayesClassifier):
    """
    Extended Naive Bayes classifier with additional methods for assignment problems.
    """
    
    def __init__(self):
        super().__init__()
        self.word_counts = {}
    
    def fit(self, X, y):
        """
        Override fit to also store word counts for smoothing.
        """
        super().fit(X, y)
        
        self.word_counts = {class_label: {} for class_label in self.classes}
        
        for class_label in self.classes:
            class_docs = [X[i] for i in range(len(X)) if y[i] == class_label]
            
            word_counts = Counter()
            for doc in class_docs:
                word_counts.update(doc)
            
            self.word_counts[class_label] = word_counts
    
    def predict_single_with_scores(self, document):
        """
        Predict a single document and return both prediction and raw scores.
        
        Args:
            document: List of tokens
            
        Returns:
            tuple: (predicted_class, scores_dict)
        """
        if not self.class_priors:
            raise ValueError("Classifier must be trained before making predictions")
        
        scores = {}
        
        for class_label in self.class_priors:
            score = math.log(self.class_priors[class_label])
            
            for word in document:
                if word in self.word_probs[class_label] and self.word_probs[class_label][word] > 0:
                    score += math.log(self.word_probs[class_label][word])
                else:
                    score += math.log(1.0 / (sum(self.word_counts[class_label].values()) + len(self.vocabulary)))
            
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
        Get probability distribution over classes for a document.
        
        Args:
            document: List of tokens
            
        Returns:
            dict: Probability distribution over classes
        """
        _, scores = self.predict_single_with_scores(document)
        
        score_values = np.array(list(scores.values()))
        class_labels = list(scores.keys())
        
        exp_scores = np.exp(score_values - np.max(score_values))
        probabilities_array = exp_scores / np.sum(exp_scores)
        
        probabilities = {class_labels[i]: probabilities_array[i] for i in range(len(class_labels))}
        return probabilities
    
    def get_most_predictive_words(self, n_words=10):
        """
        Find the most predictive words for each class using NumPy for efficiency.
        
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
                if word in self.word_probs[class_label]:
                    class_prob = self.word_probs[class_label][word]
                    
                    if class_prob > 0:
                        other_probs = []
                        for other_class in self.class_priors:
                            if other_class != class_label and word in self.word_probs[other_class]:
                                other_prob = self.word_probs[other_class][word]
                                if other_prob > 0:
                                    other_probs.append(other_prob)
                        
                        if other_probs:
                            avg_other_prob = np.mean(other_probs)
                            score = np.log(class_prob) - np.log(avg_other_prob)
                            word_scores.append((word, score))
            
            word_scores.sort(key=lambda x: x[1], reverse=True)
            predictive_words[class_label] = word_scores[:n_words]
        
        return predictive_words


def problem_1_worst_movie():
    """
    Problem 1: Find the most "rotten" movie review in the test set.
    """
    print("=" * 60)
    print("PROBLEM 1: Finding the Most Rotten Movie Review")
    print("=" * 60)
    
    train_path = os.path.join(SCRIPT_DIR, "data_rt_train.csv")
    test_path = os.path.join(SCRIPT_DIR, "data_rt_test.csv")
    
    X_train, y_train = load_and_process_data(train_path)
    X_test, y_test = load_and_process_data(test_path)
    
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    predictions, scores_list = classifier.predict_with_scores(X_test)
    
    min_rotten_score = float('inf')
    worst_review_idx = -1
    
    for i, scores in enumerate(scores_list):
        rotten_score = scores['0']
        if rotten_score < min_rotten_score:
            min_rotten_score = rotten_score
            worst_review_idx = i
    
    with open(test_path, 'r', encoding='utf-8') as file:
        test_lines = file.readlines()
    
    worst_review_line = test_lines[worst_review_idx].strip()
    actual_label, review_text = worst_review_line.split(',', 1)
    
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
    
    train_path = os.path.join(SCRIPT_DIR, "data_rt_train.csv")
    X_train, y_train = load_and_process_data(train_path)
    
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    jurassic_review = "Jurassic Park is a cautionary tale about science gone wrong and filmmaking gone lazy. For all its groundbreaking effects, the plot is held together with dino-sized leaps in logic and characters who make decisions so dumb they deserve to be eaten. The kids are annoying, the adults are incompetent, and Jeff Goldblum spends half the movie shirtless and smirking like he's in a cologne ad. It's less a thrilling adventure and more a theme park ride with a script written on the back of a napkin."
    
    tokenized_review = tokenize_text(jurassic_review)
    
    prob_dist = classifier.get_probability_distribution(tokenized_review)
    
    print("Jurassic Park Review Analysis:")
    print(f"Review length: {len(jurassic_review)} characters, {len(tokenized_review)} tokens")
    print(f"\nProbability Distribution:")
    print(f"P(Rotten|review) = {prob_dist['0']:.4f}")
    print(f"P(Fresh|review) = {prob_dist['1']:.4f}")
    
    prediction, _ = classifier.predict_single_with_scores(tokenized_review)
    print(f"\nPredicted Class: {prediction} ({'Rotten' if prediction == '0' else 'Fresh'})")
    
    _, raw_scores = classifier.predict_single_with_scores(tokenized_review)
    print(f"Raw log scores - Rotten: {raw_scores['0']:.4f}, Fresh: {raw_scores['1']:.4f}")
    print()


def problem_3_roc_curve():
    """
    Problem 3: Generate ROC curve.
    """
    print("=" * 60)
    print("PROBLEM 3: ROC Curve Generation")
    print("=" * 60)
    
    train_path = os.path.join(SCRIPT_DIR, "data_rt_train.csv")
    test_path = os.path.join(SCRIPT_DIR, "data_rt_test.csv")
    
    X_train, y_train = load_and_process_data(train_path)
    X_test, y_test = load_and_process_data(test_path)
    
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    _, scores_list = classifier.predict_with_scores(X_test)
    
    fresh_scores = np.array([scores['1'] for scores in scores_list])
    
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
    plt.show()
    
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
    
    train_path = os.path.join(SCRIPT_DIR, "data_rt_train.csv")
    X_train, y_train = load_and_process_data(train_path)
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    predictive_words = classifier.get_most_predictive_words(10)
    
    print("10 Most Indicative Words for ROTTEN reviews (class '0'):")
    for i, (word, score) in enumerate(predictive_words['0'], 1):
        print(f"{i:2d}. {word:15s} (score: {score:8.4f})")
    
    print("\n10 Most Indicative Words for FRESH reviews (class '1'):")
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
    
    train_path = os.path.join(SCRIPT_DIR, "data_rt_train.csv")
    X_train, y_train = load_and_process_data(train_path)
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    predictive_words = classifier.get_most_predictive_words(20)
    rotten_words = [word for word, _ in predictive_words['0']]
    
    print("Most rotten-indicative words to use:", rotten_words[:15])
    
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
    print(f'"{deceptive_review}"')
    print(f"\nTokenized: {tokenized_review}")
    print(f"\nClassifier Results:")
    print(f"Predicted Class: {prediction} ({'Rotten' if prediction == '0' else 'Fresh'})")
    print(f"Probability Distribution:")
    print(f"  P(Rotten|review) = {prob_dist['0']:.4f}")
    print(f"  P(Fresh|review) = {prob_dist['1']:.4f}")
    print(f"Raw log scores - Rotten: {raw_scores['0']:.4f}, Fresh: {raw_scores['1']:.4f}")
    print()


def main():
    """
    Main function to run all problems.
    """
    
    problem_1_worst_movie()
    problem_2_jurassic_park()
    problem_3_roc_curve()
    problem_4_predictive_words()
    problem_5_deceptive_review()

if __name__ == "__main__":
    main()