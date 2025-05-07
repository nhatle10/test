import argparse
import sys
import os
import pandas as pd
import numpy as np
# --- Import Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
# from catboost import CatBoostClassifier # Uncomment if you have catboost installed

# --- Import Preprocessing & Pipeline ---
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from joblib import dump, load
import time

# Define ROC AUC scorer
roc_auc = make_scorer(roc_auc_score)

def run_train(public_dir, model_dir):
    """
    Loads training data, performs Grid Search across different models
    and parameters with preprocessing, and saves the best found pipeline.
    """
    os.makedirs(model_dir, exist_ok=True)

    print("Loading training data...")
    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    try:
        df = pd.read_json(train_path, lines=True)
    except FileNotFoundError:
        print(f"Error: Training data not found at {train_path}")
        sys.exit(1)
    except Exception as e:
         print(f"Error loading training data: {e}")
         sys.exit(1)


    print(f"Training data loaded. Shape: {df.shape}")

    # Drop the target variable and potentially identifier columns
    if 'two_year_recid' not in df.columns:
        print("Error: Target column 'two_year_recid' not found in training data.")
        sys.exit(1)

    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    # Identify categorical and numerical columns
    # Ensure column names are strings
    X.columns = X.columns.astype(str)
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Identified {len(num_features)} numerical features and {len(cat_features)} categorical features.")

    # --- Preprocessing Pipeline Definition ---
    # Numerical features: Impute with mean, then Scale
    # Categorical features: Impute with most frequent, then One-Hot Encode
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([ # Pipeline for numerical features
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()) # Added StandardScaler
            ]), num_features),
            ('cat', Pipeline([ # Pipeline for categorical features
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_features)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # --- Full Pipeline with Preprocessing and Model Placeholder ---
    # We use 'classifier' as a placeholder step name that GridSearchCV will replace
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression())]) # Initial placeholder model

    # --- Parameter Grid for Grid Search ---
    # This defines the different models and their hyperparameters to search over.
    # Each dictionary in the list represents a specific model type and its parameter grid.
    # WARNING: Expanding the grid significantly increases training time.
    # Start with a smaller grid and expand if needed.
    param_grid = [
        # 1. Logistic Regression
        {
            'classifier': [LogisticRegression(max_iter=2000, random_state=42)], # Increased max_iter
            'classifier__solver': ['liblinear', 'lbfgs'], # Common solvers
            'classifier__penalty': ['l2'], # lbfgs only supports l2 or None
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0], # Regularization strength
        },
        # Add 'l1' penalty option if using liblinear/saga solver separately
        {
            'classifier': [LogisticRegression(max_iter=2000, random_state=42)],
            'classifier__solver': ['liblinear'],
            'classifier__penalty': ['l1'],
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
        },

        # 2. KNeighborsClassifier
        {
            'classifier': [KNeighborsClassifier()],
            'classifier__n_neighbors': [3, 5, 7, 10, 15], # Number of neighbors
            'classifier__weights': ['uniform', 'distance'], # Weight function
            'classifier__p': [1, 2], # Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
        },

        # 3. Decision Tree Classifier
        {
            'classifier': [DecisionTreeClassifier(random_state=42)],
            'classifier__criterion': ['gini', 'entropy'], # Split criterion
            'classifier__max_depth': [5, 10, 20, None], # Maximum depth
            'classifier__min_samples_split': [2, 5, 10], # Minimum samples to split a node
            'classifier__min_samples_leaf': [1, 2, 4], # Minimum samples required at a leaf node
            'classifier__max_features': ['sqrt', 'log2', None], # Number of features to consider for best split
        },

        # 4. Random Forest Classifier
        {
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [100, 200, 300, 500], # Number of trees
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__bootstrap': [True, False], # Whether bootstrap samples are used
        },

        # 5. Gradient Boosting Classifier
        {
             'classifier': [GradientBoostingClassifier(random_state=42)],
             'classifier__n_estimators': [100, 200, 300],
             'classifier__learning_rate': [0.1, 0.05, 0.01],
             'classifier__max_depth': [3, 5, 7],
             'classifier__min_samples_split': [2, 5],
             'classifier__min_samples_leaf': [1, 2],
             'classifier__subsample': [0.8, 1.0], # Subsample of the training instances
             'classifier__max_features': ['sqrt', None],
        },
        # 6. AdaBoost Classifier
        {
             'classifier': [AdaBoostClassifier(random_state=42)],
             'classifier__n_estimators': [50, 100, 200],
             'classifier__learning_rate': [1.0, 0.5, 0.1, 0.01],
             # base_estimator is typically DecisionTreeClassifier(max_depth=1), often not tuned here
             # 'classifier__base_estimator': [DecisionTreeClassifier(max_depth=1)], # Example if you wanted to tune base_estimator
        },

        # 7. Gaussian Naive Bayes
        # Only one significant parameter to tune
        {
            'classifier': [GaussianNB()],
            'classifier__var_smoothing': [1e-9, 1e-10, 1e-11], # Additive (Laplace/Lidstone) smoothing parameter
        },

        # 8. MLP Classifier (Neural Network)
        # Tuning MLP can be complex and sensitive. Fewer options here.
        {
            'classifier': [MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)], # Increased max_iter, use early_stopping
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50,), (100, 50,)], # Different network structures
            'classifier__activation': ['relu', 'tanh'], # Activation function
            'classifier__solver': ['adam'], # Adam is often a good default
            'classifier__alpha': [0.0001, 0.001, 0.01], # L2 penalty (regularization)
            'classifier__learning_rate': ['constant', 'adaptive'], # Learning rate schedule
        },

        # 9. XGBoost Classifier
        {
            'classifier': [xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.1, 0.05, 0.01],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.8, 1.0], # Fraction of samples used for fitting the individual base learners
            'classifier__colsample_bytree': [0.8, 1.0], # Fraction of columns when constructing each tree
            'classifier__gamma': [0, 0.1, 0.5], # Minimum loss reduction required to make a further partition
            'classifier__reg_alpha': [0, 0.1, 0.5], # L1 regularization
            'classifier__reg_lambda': [1, 0.1, 0.5], # L2 regularization
        },

        # 10. LightGBM Classifier
        {
            'classifier': [lgb.LGBMClassifier(random_state=42)],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.1, 0.05, 0.01],
            'classifier__num_leaves': [31, 64, 128], # Max number of leaves in one tree (main parameter for complexity)
            'classifier__max_depth': [-1, 10, 20], # Max depth (-1 means no limit)
            'classifier__min_child_samples': [20, 50], # Minimum number of data needed in a child
            'classifier__subsample': [0.8, 1.0], # Fraction of samples
            'classifier__colsample_bytree': [0.8, 1.0], # Fraction of features
            'classifier__reg_alpha': [0, 0.1, 0.5], # L1 regularization
            'classifier__reg_lambda': [0, 0.1, 0.5], # L2 regularization
        },

        # 11. CatBoost Classifier (Uncomment if installed)
        # {
        #     'classifier': [CatBoostClassifier(verbose=0, random_state=42, iterations=300)], # verbose=0, fixed iterations for search
        #     'classifier__learning_rate': [0.1, 0.05, 0.01],
        #     'classifier__depth': [4, 6, 8],
        #     'classifier__l2_leaf_reg': [3, 1, 5], # L2 regularization
        #     'classifier__border_count': [32, 64, 128], # For numerical features binning
        #     # cat_features parameter is handled by the pipeline's OneHotEncoder BEFORE CatBoost sees the data.
        #     # If using CatBoost's internal categorical handling, preprocessor needs adjustment.
        #     # Assuming OHE is sufficient for this example.
        # },

        # 12. SVC (Support Vector Classifier) - Can be very slow on large datasets
        # Consider reducing the C and gamma ranges or commenting this out if time is an issue.
        {
             'classifier': [SVC(probability=True, random_state=42)], # probability=True is needed for roc_auc_score with .predict()
             'classifier__C': [0.1, 1.0, 10.0], # Regularization parameter
             'classifier__kernel': ['rbf'], # RBF is common, 'linear' might also be tried
             'classifier__gamma': ['scale', 'auto', 0.1, 1.0], # Kernel coefficient
             # 'classifier__degree': [3], # For 'poly' kernel
        },
        # Add LinearSVC as a potentially faster linear alternative to SVC(kernel='linear')
        # {
        #     'classifier': [LinearSVC(max_iter=2000, random_state=42)],
        #     'classifier__penalty': ['l2'],
        #     'classifier__C': [0.1, 1.0, 10.0],
        #     # Note: LinearSVC does not have predict_proba, so it cannot be used directly with roc_auc_score
        #     # unless you switch to a score that uses predict (like accuracy) or wrap it.
        #     # Sticking to SVC(probability=True) for consistency with roc_auc.
        # }
    ]

    # --- Set up Grid Search ---
    # cv=5 means 5-fold cross-validation
    # scoring='roc_auc' uses the ROC AUC score to evaluate models
    # n_jobs=-1 uses all available CPU cores (set to a smaller number if needed, e.g., 4)
    print("Starting Grid Search with expanded parameter grid...")
    # Calculate total number of candidates (this might be large!)
    total_candidates = 0
    for d in param_grid:
        # For each dict, get the list of classifiers (the first item)
        classifiers_list = list(d.values())[0]
        # Count other parameter combinations
        other_params = {k: v for k, v in d.items() if k != 'classifier'}
        num_other_combinations = np.prod([len(v) for v in other_params.values()]) if other_params else 1
        total_candidates += len(classifiers_list) * num_other_combinations

    print(f"Total number of candidate configurations: {total_candidates}")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=roc_auc, n_jobs=-1, verbose=3) # Increased verbose for more output

    start_time = time.time()
    # Fit on the entire training data (GridSearchCV handles internal splitting for CV)
    grid_search.fit(X, y)
    end_time = time.time()

    print("\nGrid Search finished.")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # Get the best model found by Grid Search
    best_pipeline = grid_search.best_estimator_

    print("\nBest parameters found:")
    # Accessing best_params_ will show the entire parameter setting for the best pipeline,
    # including which classifier was chosen and its params.
    print(grid_search.best_params_)
    print("\nBest cross-validation ROC AUC score:")
    print(grid_search.best_score_)

    # Save the entire best pipeline
    model_path = os.path.join(model_dir, 'best_pipeline.joblib')
    print(f"\nSaving best pipeline to {model_path}")
    try:
        dump(best_pipeline, model_path)
        print("Training complete. Best pipeline saved.")
    except Exception as e:
        print(f"Error saving pipeline: {e}")
        # Continue running to see search results even if save fails


def run_predict(model_dir, test_input_dir, output_path):
    """
    Loads the best trained pipeline and makes predictions on test data.
    """
    # Load the best pipeline
    model_path = os.path.join(model_dir, 'best_pipeline.joblib')
    print(f"Loading pipeline from {model_path}")
    try:
        pipeline = load(model_path)
        print("Pipeline loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Pipeline file not found at {model_path}")
        sys.exit(1)
    except Exception as e:
         print(f"Error loading pipeline: {e}")
         sys.exit(1)

    # Load test data from JSON
    test_path = os.path.join(test_input_dir, 'test.json')
    print(f"Loading test data from {test_path}")
    try:
        df_test = pd.read_json(test_path, lines=True)
        print(f"Test data loaded. Shape: {df_test.shape}")
    except FileNotFoundError:
        print(f"Error: Test data not found at {test_path}")
        sys.exit(1)
    except Exception as e:
         print(f"Error loading test data: {e}")
         sys.exit(1)

    # Ensure column names are strings to match training columns after preprocessing setup
    df_test.columns = df_test.columns.astype(str)

    # Make predictions
    # The pipeline automatically handles preprocessing and prediction
    print("Making predictions...")
    try:
        # Use predict() for class labels as required by the original output format
        preds = pipeline.predict(df_test)
        print("Predictions made.")
        # If you needed probabilities for evaluation (like ROC AUC on a holdout set), use:
        # probs = pipeline.predict_proba(df_test)[:, 1] # for positive class

    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


    # Save predictions with proper column name
    print(f"Saving predictions to {output_path}")
    try:
        pd.DataFrame({'two_year_recid': preds}).to_json(output_path, orient='records', lines=True)
        print("Predictions saved.")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Machine Learning script for recidivism prediction with Grid Search across multiple models and parameters.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    parser_train = subparsers.add_parser('train', help='Train the model using Grid Search')
    parser_train.add_argument('--public_dir', type=str, required=True, help='Directory containing the training data (e.g., public/data). Expects data at public_dir/train_data/train.json')
    parser_train.add_argument('--model_dir', type=str, required=True, help='Directory to save the trained model pipeline')

    # Predict command
    parser_predict = subparsers.add_parser('predict', help='Make predictions using a trained model')
    parser_predict.add_argument('--model_dir', type=str, required=True, help='Directory containing the trained model pipeline')
    parser_predict.add_argument('--test_input_dir', type=str, required=True, help='Directory containing the test input data. Expects data at test_input_dir/test.json')
    # --- CORRECTED LINE BELOW ---
    parser_predict.add_argument('--output_path', type=str, required=True, help='File path to save the predictions (JSON format)')

    args = parser.parse_args()

    # --- Path Validation and Directory Creation (after parsing args) ---
    if args.command == 'train':
        # Check if model_dir parent exists or can be created
        if args.model_dir: # Check if argument was provided (should be true due to required=True, but defensive)
             model_dir_parent = os.path.dirname(args.model_dir)
             if model_dir_parent and not os.path.exists(model_dir_parent):
                  try:
                      os.makedirs(model_dir_parent, exist_ok=True)
                      print(f"Created directory: {model_dir_parent}")
                  except OSError as e:
                      print(f"Error creating model directory parent {model_dir_parent}: {e}")
                      sys.exit(1)

        run_train(args.public_dir, args.model_dir)

    elif args.command == 'predict':
         # Check if test_input_dir exists
         if not os.path.exists(args.test_input_dir):
              print(f"Error: Test input directory not found: {args.test_input_dir}")
              sys.exit(1)

         # Check if output directory exists and create it if necessary
         output_dir = os.path.dirname(args.output_path)
         if output_dir and not os.path.exists(output_dir):
             try:
                 os.makedirs(output_dir, exist_ok=True)
                 print(f"Created output directory: {output_dir}")
             except OSError as e:
                 print(f"Error creating output directory {output_dir}: {e}")
                 sys.exit(1)

         run_predict(args.model_dir, args.test_input_dir, args.output_path)
    else:
        parser.print_help()
        sys.exit(1)

# Removed the unnecessary add_path_argument helper function definition

if __name__ == "__main__":
    main()