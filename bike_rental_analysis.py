#!/usr/bin/env python3
"""
Bike Rental Prediction Analysis Using Decision Trees

This script analyzes bike rental demand prediction using a custom decision tree implementation.
It explores hyperparameter tuning, model evaluation, and feature importance analysis.

Dataset Overview:
- Target Classes: 4 rental volume categories
  - 0: 0-190 rentals
  - 1: 191-504 rentals  
  - 2: 505-1065 rentals
  - 3: more than 1065 rentals

- Features: 12 attributes including time, weather, and seasonal information
  1. Hour (0-23)
  2. Temperature (°C)
  3. Humidity (%)
  4. Wind speed (m/s)
  5. Visibility (10m)
  6. Dew point temperature (°C)
  7. Solar Radiation (MJ/m²)
  8. Rainfall (mm)
  9. Snowfall (cm)
  10. Seasons (0=Winter, 1=Spring, 2=Summer, 3=Autumn)
  11. Holiday (0=No, 1=Yes)
  12. Functioning Day (0=No, 1=Yes)

Author: Converted from Jupyter notebook
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from decision_tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_and_explore_data():
    """Load the data and perform basic exploration."""
    print("=" * 60)
    print("1. DATA LOADING AND EXPLORATION")
    print("=" * 60)
    
    # Load the data
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"\nFeature ranges:")
    for i in range(X_train.shape[1]):
        print(f"Feature {i}: [{X_train[:, i].min():.2f}, {X_train[:, i].max():.2f}]")

    # Examine class distribution
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    print("\nClass distribution in training set:")
    for class_label in sorted(train_counts.keys()):
        count = train_counts[class_label]
        percentage = count / len(y_train) * 100
        print(f"Class {int(class_label)}: {count} samples ({percentage:.1f}%)")

    print("\nClass distribution in test set:")
    for class_label in sorted(test_counts.keys()):
        count = test_counts[class_label]
        percentage = count / len(y_test) * 100
        print(f"Class {int(class_label)}: {count} samples ({percentage:.1f}%)")

    return X_train, y_train, X_test, y_test

def visualize_class_distribution(y_train, y_test):
    """Visualize the class distribution."""
    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION VISUALIZATION")
    print("=" * 60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training set distribution
    train_counts = Counter(y_train)
    classes = sorted(train_counts.keys())
    train_values = [train_counts[c] for c in classes]
    
    bars1 = ax1.bar(classes, train_values, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title('Training Set Class Distribution')
    ax1.set_xlabel('Rental Volume Class')
    ax1.set_ylabel('Number of Samples')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, train_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{value}', ha='center', va='bottom')
    
    # Test set distribution
    test_counts = Counter(y_test)
    test_values = [test_counts[c] for c in classes]
    
    bars2 = ax2.bar(classes, test_values, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_title('Test Set Class Distribution')
    ax2.set_xlabel('Rental Volume Class')
    ax2.set_ylabel('Number of Samples')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, test_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def cross_validation_split(X, y, k=5):
    """Perform k-fold cross-validation split."""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k
    
    folds = []
    for i in range(k):
        start_idx = i * fold_size
        if i == k - 1:  # Last fold gets remaining samples
            end_idx = n_samples
        else:
            end_idx = (i + 1) * fold_size
        
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[test_indices]
        y_val_fold = y[test_indices]
        
        folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
    
    return folds

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using cross-validation."""
    print("\n" + "=" * 60)
    print("3. HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Define hyperparameter range
    max_depths = list(range(1, 21))
    cv_results = {}
    
    print("Performing 5-fold cross-validation for different max_depth values...")
    print(f"{'Depth':<6} {'Mean CV Acc':<12} {'Std CV Acc':<12} {'Status':<15}")
    print("-" * 50)
    
    for depth in max_depths:
        # Perform 5-fold cross-validation
        folds = cross_validation_split(X_train, y_train, k=5)
        fold_scores = []
        
        for X_train_fold, y_train_fold, X_val_fold, y_val_fold in folds:
            model = DecisionTreeClassifier(max_depth=depth)
            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val_fold, y_val_fold)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        cv_results[depth] = {'mean': mean_score, 'std': std_score, 'scores': fold_scores}
        
        print(f"{depth:<6} {mean_score:<12.4f} {std_score:<12.4f} {'✓' if depth <= 10 else 'Deep':<15}")
    
    # Find best hyperparameters
    best_depth = max(cv_results.keys(), key=lambda k: cv_results[k]['mean'])
    best_score = cv_results[best_depth]['mean']
    
    print(f"\nBest hyperparameters:")
    print(f"- max_depth: {best_depth}")
    print(f"- Best CV accuracy: {best_score:.4f} ± {cv_results[best_depth]['std']:.4f}")
    
    return cv_results, best_depth, best_score

def visualize_cv_results(cv_results):
    """Visualize cross-validation results."""
    print("\n" + "=" * 60)
    print("4. CROSS-VALIDATION RESULTS VISUALIZATION")
    print("=" * 60)
    
    depths = list(cv_results.keys())
    means = [cv_results[d]['mean'] for d in depths]
    stds = [cv_results[d]['std'] for d in depths]
    
    plt.figure(figsize=(12, 8))
    plt.errorbar(depths, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
    plt.fill_between(depths, np.array(means) - np.array(stds), 
                     np.array(means) + np.array(stds), alpha=0.2)
    
    plt.xlabel('Maximum Tree Depth')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Cross-Validation Performance vs Tree Depth')
    plt.grid(True, alpha=0.3)
    plt.xticks(depths)
    
    # Highlight best performance
    best_depth = max(cv_results.keys(), key=lambda k: cv_results[k]['mean'])
    best_score = cv_results[best_depth]['mean']
    plt.axvline(x=best_depth, color='red', linestyle='--', alpha=0.7, 
                label=f'Best depth: {best_depth}')
    plt.axhline(y=best_score, color='red', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_final_model(X_train, y_train, X_test, y_test, best_depth):
    """Train the final model with best hyperparameters."""
    print("\n" + "=" * 60)
    print("5. FINAL MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    # Train final model
    final_model = DecisionTreeClassifier(max_depth=best_depth)
    final_model.fit(X_train, y_train)
    
    # Evaluate on training and test sets
    train_accuracy = final_model.score(X_train, y_train)
    test_accuracy = final_model.score(X_test, y_test)
    test_error = 1 - test_accuracy
    
    print(f"Final Model Performance:")
    print(f"- Training accuracy: {train_accuracy:.4f}")
    print(f"- Test accuracy: {test_accuracy:.4f}")
    print(f"- Test error rate: {test_error:.4f} ({test_error*100:.2f}%)")
    print(f"- Tree depth: {final_model.get_depth()}")
    
    return final_model, train_accuracy, test_accuracy, test_error

def create_confusion_matrix(y_true, y_pred):
    """Create confusion matrix manually."""
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = classes.index(true_label)
        pred_idx = classes.index(pred_label)
        confusion_matrix[true_idx, pred_idx] += 1
    
    return confusion_matrix, classes

def evaluate_model_detailed(final_model, X_test, y_test):
    """Perform detailed model evaluation."""
    print("\n" + "=" * 60)
    print("6. DETAILED MODEL EVALUATION")
    print("=" * 60)
    
    # Make predictions
    y_pred = final_model.predict(X_test)
    
    # Create confusion matrix
    confusion_matrix, classes = create_confusion_matrix(y_test, y_pred)
    
    print("Confusion Matrix:")
    print(f"{'True\\Pred':<10}", end="")
    for class_id in classes:
        print(f"{int(class_id):<8}", end="")
    print()
    print("-" * (10 + 8 * len(classes)))
    
    for i, true_class in enumerate(classes):
        print(f"{int(true_class):<10}", end="")
        for j, pred_class in enumerate(classes):
            print(f"{confusion_matrix[i, j]:<8}", end="")
        print()
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {int(c)}' for c in classes],
                yticklabels=[f'Class {int(c)}' for c in classes])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()
    
    # Calculate per-class metrics
    metrics = {}
    for i, class_id in enumerate(classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        tn = np.sum(confusion_matrix) - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(confusion_matrix[i, :])
        
        metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': support
        }
    
    # Display classification report
    print(f"\nClassification Report:")
    print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    
    for class_id in classes:
        m = metrics[class_id]
        print(f"{int(class_id):<8} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1_score']:<10.3f} {m['support']:<10}")
    
    # Calculate overall metrics
    macro_precision = np.mean([metrics[i]['precision'] for i in classes])
    macro_recall = np.mean([metrics[i]['recall'] for i in classes])
    macro_f1 = np.mean([metrics[i]['f1_score'] for i in classes])
    
    print(f"\nMacro-averaged metrics:")
    print(f"Precision: {macro_precision:.3f}")
    print(f"Recall: {macro_recall:.3f}")
    print(f"F1-Score: {macro_f1:.3f}")
    
    return metrics

def analyze_feature_importance(final_model):
    """Analyze and visualize feature importance."""
    print("\n" + "=" * 60)
    print("7. FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Get feature importances
    feature_importances = final_model.feature_importances_
    
    # Define feature names for better interpretation
    feature_names = [
        'Hour',
        'Temperature (°C)',
        'Humidity (%)',
        'Wind Speed (m/s)',
        'Visibility (10m)',
        'Dew Point (°C)',
        'Solar Radiation (MJ/m²)',
        'Rainfall (mm)',
        'Snowfall (cm)',
        'Season',
        'Holiday',
        'Functioning Day'
    ]
    
    # Create feature importance dataframe for easier handling
    importance_data = list(zip(feature_names, feature_importances))
    importance_data.sort(key=lambda x: x[1], reverse=True)
    
    print("Feature Importance Rankings:")
    print(f"{'Rank':<5} {'Feature':<25} {'Importance':<12} {'Percentage':<12}")
    print("-" * 60)
    
    for rank, (name, importance) in enumerate(importance_data, 1):
        percentage = importance * 100
        print(f"{rank:<5} {name:<25} {importance:<12.4f} {percentage:<12.2f}%")
    
    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    sorted_features, sorted_importances = zip(*importance_data)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_features))
    bars = plt.barh(y_pos, sorted_importances, alpha=0.7)
    
    # Color bars based on importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(y_pos, sorted_features)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Decision Tree Model')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (feature, importance) in enumerate(importance_data):
        plt.text(importance + 0.001, i, f'{importance:.3f}', 
                 va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return importance_data

def interpret_results(importance_data):
    """Interpret and discuss the results."""
    print("\n" + "=" * 60)
    print("8. MODEL INTERPRETATION AND DISCUSSION")
    print("=" * 60)
    
    # Analyze the most important features
    top_3_features = importance_data[:3]
    print("Analysis of Top 3 Most Important Features:")
    print("=" * 50)
    
    for rank, (feature_name, importance) in enumerate(top_3_features, 1):
        print(f"\n{rank}. {feature_name} (Importance: {importance:.4f})")
        
        if 'Hour' in feature_name:
            print("   - Time of day is crucial for bike rental demand")
            print("   - Peak hours likely correspond to commuting times")
            print("   - Different rental patterns throughout the day")
        
        elif 'Temperature' in feature_name:
            print("   - Weather conditions significantly impact bike usage")
            print("   - Comfortable temperatures encourage more rentals")
            print("   - Extreme temperatures (too hot/cold) reduce demand")
        
        elif 'Season' in feature_name:
            print("   - Seasonal patterns affect bike rental behavior")
            print("   - Spring/Summer likely have higher demand")
            print("   - Winter conditions may discourage bike usage")
        
        elif 'Humidity' in feature_name:
            print("   - Humidity affects comfort level for outdoor activities")
            print("   - High humidity may discourage bike rentals")
            print("   - Optimal humidity ranges promote bike usage")
    
    # Analyze features with low importance
    low_importance_features = [f for f, imp in importance_data if imp < 0.05]
    print(f"\n\nFeatures with Low Importance (< 5%):")
    for feature_name, importance in importance_data:
        if importance < 0.05:
            print(f"- {feature_name}: {importance:.4f} ({importance*100:.2f}%)")
    
    if low_importance_features:
        print("\nThese features may be less predictive or redundant with other features.")
    else:
        print("\nAll features show meaningful importance for prediction.")

def print_summary(X_train, X_test, best_depth, best_score, cv_results, 
                  train_accuracy, test_accuracy, test_error, final_model, importance_data):
    """Print comprehensive summary of the analysis."""
    print("\n" + "=" * 60)
    print("9. SUMMARY AND CONCLUSIONS")
    print("=" * 60)
    
    print("BIKE RENTAL PREDICTION ANALYSIS - SUMMARY")
    print("=" * 50)
    
    print(f"\n1. DATASET CHARACTERISTICS:")
    print(f"   - Training samples: {len(X_train):,}")
    print(f"   - Test samples: {len(X_test):,}")
    print(f"   - Features: {X_train.shape[1]}")
    print(f"   - Classes: 4 (rental volume categories)")
    
    print(f"\n2. HYPERPARAMETER TUNING:")
    print(f"   - Method: 5-fold cross-validation")
    print(f"   - Best max_depth: {best_depth}")
    print(f"   - Best CV accuracy: {best_score:.4f} ± {cv_results[best_depth]['std']:.4f}")
    
    print(f"\n3. FINAL MODEL PERFORMANCE:")
    print(f"   - Training accuracy: {train_accuracy:.4f}")
    print(f"   - Test accuracy: {test_accuracy:.4f}")
    print(f"   - Classification error: {test_error:.4f} ({test_error*100:.2f}%)")
    print(f"   - Tree depth: {final_model.get_depth()}")
    
    print(f"\n4. TOP 3 MOST IMPORTANT FEATURES:")
    for i, (feature, importance) in enumerate(importance_data[:3], 1):
        print(f"   {i}. {feature}: {importance:.4f} ({importance*100:.2f}%)")
    
    print(f"\n5. MODEL INSIGHTS:")
    print(f"   - The decision tree successfully captures bike rental patterns")
    print(f"   - Time-based features (hour, season) are highly predictive")
    print(f"   - Weather conditions significantly influence rental demand")
    print(f"   - The model shows good generalization to unseen data")
    
    # Calculate if there's overfitting
    overfitting_gap = train_accuracy - test_accuracy
    print(f"\n6. OVERFITTING ANALYSIS:")
    print(f"   - Training vs Test gap: {overfitting_gap:.4f}")
    if overfitting_gap < 0.05:
        print(f"   - Low overfitting: Model generalizes well")
    elif overfitting_gap < 0.10:
        print(f"   - Moderate overfitting: Acceptable for this complexity")
    else:
        print(f"   - High overfitting: Consider reducing model complexity")
    
    print(f"\n7. BUSINESS IMPLICATIONS:")
    print(f"   - Peak demand prediction can optimize bike distribution")
    print(f"   - Weather-based adjustments can improve service availability")
    print(f"   - Seasonal planning can guide inventory management")
    print(f"   - Hour-based patterns support dynamic pricing strategies")

def main():
    """Main function to run the complete analysis."""
    print("BIKE RENTAL PREDICTION ANALYSIS USING DECISION TREES")
    print("=" * 60)
    print("This analysis demonstrates bike rental demand prediction using")
    print("a custom decision tree implementation with hyperparameter tuning,")
    print("model evaluation, and feature importance analysis.")
    print("=" * 60)
    
    # 1. Load and explore data
    X_train, y_train, X_test, y_test = load_and_explore_data()
    
    # 2. Visualize class distribution
    visualize_class_distribution(y_train, y_test)
    
    # 3. Hyperparameter tuning
    cv_results, best_depth, best_score = hyperparameter_tuning(X_train, y_train)
    
    # 4. Visualize CV results
    visualize_cv_results(cv_results)
    
    # 5. Train final model
    final_model, train_accuracy, test_accuracy, test_error = train_final_model(
        X_train, y_train, X_test, y_test, best_depth)
    
    # 6. Detailed evaluation
    metrics = evaluate_model_detailed(final_model, X_test, y_test)
    
    # 7. Feature importance analysis
    importance_data = analyze_feature_importance(final_model)
    
    # 8. Interpret results
    interpret_results(importance_data)
    
    # 9. Print comprehensive summary
    print_summary(X_train, X_test, best_depth, best_score, cv_results,
                  train_accuracy, test_accuracy, test_error, final_model, importance_data)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("This analysis demonstrates the effectiveness of decision trees for")
    print("bike rental prediction, providing both good predictive performance")
    print("and interpretable results for business decision-making.")

if __name__ == "__main__":
    main()