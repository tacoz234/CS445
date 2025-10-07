#!/usr/bin/env python3
"""
Naive Bayes Classifier Assignment Solutions
CS445 PA2 - Cole Determan
"""

import csv
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import roc_curve, auc
from naive_bayes import NaiveBayesClassifier, tokenize_text, load_and_process_data


class ExtendedNaiveBayesClassifier(NaiveBayesClassifier):
    """
    Extended Naive Bayes classifier with additional methods for assignment problems.
    """
    
    def predict_single_with_scores(self, document):
        """
        Predict the class for a single document and return both prediction and scores.
        
        Args:
            document (list): Tokenized document (list of words)
            
        Returns:
            tuple: (predicted_class, class_scores_dict)
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

        # Return class with highest score and all scores
        predicted_class = max(class_scores, key=class_scores.get)
        return predicted_class, class_scores
    
    def predict_with_scores(self, X):
        """
        Predict classes for multiple documents and return scores.
        
        Args:
            X (list): List of tokenized documents
            
        Returns:
            tuple: (predictions, scores_list)
        """
        predictions = []
        scores_list = []
        
        for doc in X:
            pred, scores = self.predict_single_with_scores(doc)
            predictions.append(pred)
            scores_list.append(scores)
            
        return predictions, scores_list
    
    def get_probability_distribution(self, document):
        """
        Get normalized probability distribution for a document.
        
        Args:
            document (list): Tokenized document (list of words)
            
        Returns:
            dict: Normalized probability distribution
        """
        _, log_scores = self.predict_single_with_scores(document)
        
        # Convert log scores to probabilities using softmax
        # First, subtract max for numerical stability
        max_score = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in log_scores.items()}
        
        # Normalize
        total = sum(exp_scores.values())
        probabilities = {label: score / total for label, score in exp_scores.items()}
        
        return probabilities
    
    def get_most_predictive_words(self, n_words=10):
        """
        Find the most predictive words for each class.
        
        Args:
            n_words (int): Number of words to return for each class
            
        Returns:
            dict: Dictionary with class labels as keys and lists of (word, score) tuples
        """
        # Calculate log probability ratios for each word
        word_scores = {}
        
        for word in self.vocabulary:
            # Skip words with zero probability in any class
            skip_word = False
            for class_label in self.classes:
                if self.word_probs[class_label][word] == 0:
                    skip_word = True
                    break
            
            if skip_word:
                continue
                
            # Calculate log ratio: log(P(word|class1)) - log(P(word|class0))
            # Assuming classes are '0' and '1'
            if '0' in self.classes and '1' in self.classes:
                log_ratio = (math.log(self.word_probs['1'][word]) - 
                           math.log(self.word_probs['0'][word]))
                word_scores[word] = log_ratio
        
        # Sort words by their scores
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1])
        
        # Most indicative of class '0' (rotten) - most negative scores
        rotten_words = sorted_words[:n_words]
        
        # Most indicative of class '1' (fresh) - most positive scores  
        fresh_words = sorted_words[-n_words:]
        fresh_words.reverse()  # Most positive first
        
        return {
            '0': rotten_words,  # (word, negative_score) pairs
            '1': fresh_words    # (word, positive_score) pairs
        }


def problem_1_worst_movie():
    """
    Problem 1: Find the most "rotten" movie review in the test set.
    """
    print("=" * 60)
    print("PROBLEM 1: Finding the Most Rotten Movie Review")
    print("=" * 60)
    
    # Load data
    X_train, y_train = load_and_process_data("data_rt_train.csv")
    X_test, y_test = load_and_process_data("data_rt_test.csv")
    
    # Train classifier
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    # Get predictions and scores for test set
    predictions, scores_list = classifier.predict_with_scores(X_test)
    
    # Find the review with the highest score for class '0' (rotten)
    # We want the review that the classifier is most confident is rotten
    most_rotten_idx = -1
    highest_rotten_score = float('-inf')
    
    for i, scores in enumerate(scores_list):
        rotten_score = scores['0']
        if rotten_score > highest_rotten_score:
            highest_rotten_score = rotten_score
            most_rotten_idx = i
    
    # Load original text to display
    with open("data_rt_test.csv", "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        test_reviews = list(reader)
    
    print(f"Most rotten review (index {most_rotten_idx}):")
    print(f"Actual label: {y_test[most_rotten_idx]}")
    print(f"Predicted label: {predictions[most_rotten_idx]}")
    print(f"Rotten score: {scores_list[most_rotten_idx]['0']:.4f}")
    print(f"Fresh score: {scores_list[most_rotten_idx]['1']:.4f}")
    print(f"Review text: {test_reviews[most_rotten_idx][1]}")
    print()


def problem_2_jurassic_park():
    """
    Problem 2: Analyze the Jurassic Park review.
    """
    print("=" * 60)
    print("PROBLEM 2: Jurassic Park Review Analysis")
    print("=" * 60)
    
    # Load and train
    X_train, y_train = load_and_process_data("data_rt_train.csv")
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    # The review text
    review_text = """Jurassic Park is a cautionary tale about science gone wrong and 
    filmmaking gone lazy. For all its groundbreaking effects, the plot is 
    held together with dino-sized leaps in logic and characters who make 
    decisions so dumb they deserve to be eaten. The kids are annoying, the 
    adults are incompetent, and Jeff Goldblum spends half the movie 
    shirtless and smirking like he's in a cologne ad. It's less a 
    thrilling adventure and more a theme park ride with a script written 
    on the back of a napkin."""
    
    # Tokenize the review
    tokenized_review = tokenize_text(review_text)
    
    # Get probability distribution
    prob_dist = classifier.get_probability_distribution(tokenized_review)
    
    print("Jurassic Park Review Analysis:")
    print(f"Tokenized review: {tokenized_review}")
    print(f"Probability of being ROTTEN (class '0'): {prob_dist['0']:.4f}")
    print(f"Probability of being FRESH (class '1'): {prob_dist['1']:.4f}")
    
    # Also show raw scores for reference
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
    
    # Load data
    X_train, y_train = load_and_process_data("data_rt_train.csv")
    X_test, y_test = load_and_process_data("data_rt_test.csv")
    
    # Train classifier
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    # Get scores for test set
    _, scores_list = classifier.predict_with_scores(X_test)
    
    # Extract scores for the positive class ('1' = fresh)
    # For ROC curve, we need scores where higher = more likely to be positive
    fresh_scores = [scores['1'] for scores in scores_list]
    
    # Convert string labels to integers for sklearn
    y_test_int = [int(label) for label in y_test]
    
    # Generate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_int, fresh_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
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
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"ROC curve generated and saved as 'roc_curve.png'")
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print()


def problem_4_predictive_words():
    """
    Problem 4: Find most predictive words.
    """
    print("=" * 60)
    print("PROBLEM 4: Most Predictive Words")
    print("=" * 60)
    
    # Load and train
    X_train, y_train = load_and_process_data("data_rt_train.csv")
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    # Get most predictive words
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
    
    # Load and train
    X_train, y_train = load_and_process_data("data_rt_train.csv")
    classifier = ExtendedNaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    
    # First, let's get the most predictive words to help craft our review
    predictive_words = classifier.get_most_predictive_words(20)
    rotten_words = [word for word, _ in predictive_words['0']]
    
    print("Most rotten-indicative words to use:", rotten_words[:15])
    
    # Craft a positive review using words that the classifier associates with negative reviews
    deceptive_review = """This movie is absolutely boring and tedious, but in the most 
    brilliant way possible! The plot is completely silly and stupid, yet somehow 
    manages to be pure genius. Every dull and awful moment is crafted with such 
    terrible precision that it becomes utterly magnificent. The bad acting is 
    so wonderfully horrible that it transcends into art. This mess of a film 
    is beautifully ugly and perfectly imperfect. A truly awful masterpiece 
    that fails so spectacularly it succeeds beyond measure."""
    
    # Analyze the review
    tokenized_review = tokenize_text(deceptive_review)
    prob_dist = classifier.get_probability_distribution(tokenized_review)
    prediction, raw_scores = classifier.predict_single_with_scores(tokenized_review)
    
    print("\nDeceptive Positive Review:")
    print(f'"{deceptive_review}"')
    print(f"\nTokenized: {tokenized_review}")
    print(f"\nClassifier Results:")
    print(f"Predicted class: {prediction}")
    print(f"Probability of being ROTTEN (class '0'): {prob_dist['0']:.4f}")
    print(f"Probability of being FRESH (class '1'): {prob_dist['1']:.4f}")
    print(f"Raw log scores - Rotten: {raw_scores['0']:.4f}, Fresh: {raw_scores['1']:.4f}")
    
    if prediction == '0' and prob_dist['0'] > 0.7:
        print("\n✅ SUCCESS: Positive review successfully classified as highly negative!")
    else:
        print("\n❌ Need to adjust the review to be more deceptive.")
    print()


def main():
    """
    Run all problems in the assignment.
    """
    print("CS445 PA2 - Naive Bayes Classifier Assignment")
    print("Cole Determan")
    print("=" * 60)
    
    # Run all problems
    problem_1_worst_movie()
    problem_2_jurassic_park()
    problem_3_roc_curve()
    problem_4_predictive_words()
    problem_5_deceptive_review()
    
    print("All problems completed!")


if __name__ == "__main__":
    main()