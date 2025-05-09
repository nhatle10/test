import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from joblib import dump, load
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

class NormalizeColumnByLabel(BaseEstimator, TransformerMixin):
    def __init__(self, col, label):
        self.col = col
        self.label = label
        self.transformers = {} # Dictionary to store transformers per group

    def fit(self, X, y=None):
        # Iterate through unique groups in the label column
        for group in X[self.label].unique():
            # Filter data for the current group and target column
            group_data = X.loc[X[self.label] == group, self.col]

            # --- FIX: Check if the group_data is not empty before fitting ---
            # PowerTransformer requires at least one sample to fit
            if len(group_data) > 0:
                # Initialize and fit PowerTransformer for this group
                # Reshape data to be a 2D array (-1, 1) as required by fit/transform methods
                transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                transformer.fit(group_data.values.reshape(-1, 1))
                # Store the fitted transformer
                self.transformers[group] = transformer
            else:
                # Optional: Print a warning if a group has no data
                print(f"Warning: No data found for group '{group}' in column '{self.label}' to fit PowerTransformer for column '{self.col}'. Skipping this group.")

        return self

    def transform(self, X, y=None):
        C = X.copy() # Create a copy to avoid modifying the original DataFrame

        # Iterate through unique groups in the label column of the *input* data for transformation
        for group in X[self.label].unique():
            # --- FIX: Check if a transformer was fitted for this group during the fit phase ---
            if group in self.transformers:
                # Filter data for the current group and target column in the copied DataFrame
                group_data_col = C.loc[X[self.label] == group, self.col]

                # Also ensure there is data in this group in the current batch to transform
                if len(group_data_col) > 0:
                    # Apply the fitted transformer to the group's data
                    # Reshape data to be a 2D array (-1, 1)
                    C.loc[X[self.label] == group, self.col] = self.transformers[group].transform(
                        group_data_col.values.reshape(-1, 1)
                    )
            else:
                 # Optional: Print a warning if a group exists in transform data but wasn't seen/fitted in train data
                 print(f"Warning: Group '{group}' in column '{self.label}' not found in training data or had no samples. Data for this group in column '{self.col}' will not be transformed.")

        return C
    
# from sklearn.base import BaseEstimator, TransformerMixin
# from aif360.algorithms.preprocessing.lfr import LFR
# from aif360.datasets import BinaryLabelDataset
# import pandas as pd

# class LFRCustom(BaseEstimator, TransformerMixin):
#     def __init__(self, col, protected_col, unprivileged_groups, privileged_groups):
#         self.col = col
#         self.protected_col = protected_col
#         self.unprivileged_groups = unprivileged_groups
#         self.privileged_groups = privileged_groups
#         self.TR = None

#     def fit(self, X, y=None):
#         d = pd.DataFrame(X, columns=self.col)
#         d['response'] = list(y)

#         binary_df = BinaryLabelDataset(
#             df=d,
#             protected_attribute_names=self.protected_col,
#             label_names=['response']
#         )

#         self.TR = LFR(
#             unprivileged_groups=self.unprivileged_groups,
#             privileged_groups=self.privileged_groups,
#             seed=0,
#             k=2,
#             Ax=0.5, Ay=0.2, Az=0.2,
#             verbose=0
#         )
#         self.TR.fit(binary_df, maxiter=5000, maxfun=5000)
#         return self

#     def transform(self, X, y=None):
#         d = pd.DataFrame(X, columns=self.col)
#         if y is not None:
#             d['response'] = list(y)
#         else:
#             d['response'] = False

#         binary_df = BinaryLabelDataset(
#             df=d,
#             protected_attribute_names=self.protected_col,
#             label_names=['response']
#         )

#         transformed = self.TR.transform(binary_df)
#         df_transformed = transformed.convert_to_dataframe()[0].drop(['response'], axis=1)
#         return df_transformed.values



def categorize_age(age):
    if pd.isna(age):
        return np.nan
    elif age < 25:
        return 'Less than 25'
    elif age <= 45:
        return '25 - 45'
    else:
        return 'Greater than 45'

def engineer_features(df):
    df.replace({None: np.nan}, inplace=True)

    # Ensure 'race' column exists before attempting to use it
    if 'race' in df.columns:
       # Note: This line might be problematic if 'race' is completely missing or only has NaN values
       # If it's possible for 'race' to be missing or all NaN, you might need a default 'Other' assignment
       # or imputation BEFORE this step.
       # For now, assuming 'race' exists and has at least one non-NaN value in train data.
       df.loc[df['race'].isin(['Native American', 'Asian']), 'race'] = 'Other'
    else:
       print("Warning: 'race' column not found in the DataFrame.")
       # Decide how to handle this case: create a default race column? skip race-based engineering?
       # For this fix, let's assume 'race' must exist for the NormalizeColumnByLabel step.
       # If it's truly missing, the pipeline step using 'race' will likely fail later anyway.


    # Điền giá trị NaN bằng 0 cho các cột vi phạm vị thành niên
    # Check if columns exist before filling NaNs
    juv_cols = ['juv_fel_count', 'juv_misd_count', 'juv_other_count']
    for col in juv_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            print(f"Warning: Juvenile count column '{col}' not found in the DataFrame.")
            # Add the column with zeros if missing? Or skip operations involving it?
            # For simplicity, proceeding assuming columns exist. If not, later sum will fail.


    if 'priors_count' in df.columns:
        df['priors_count'] = df['priors_count'].fillna(0)
    else:
        print("Warning: 'priors_count' column not found in the DataFrame.")
        # The NormalizeColumnByLabel step needs 'priors_count'. If missing, this will cause errors.
        # You might need to create a dummy 'priors_count' column if it's optional.

    # Total juvenile count - check if required columns exist
    if all(col in df.columns for col in juv_cols):
        df['juv_count'] = df[juv_cols].sum(axis=1)
        df = df.drop(juv_cols, axis=1)
    else:
        print("Warning: Could not calculate 'juv_count' as some juvenile columns were missing.")
        # Decide how to handle: create 'juv_count' with NaNs/zeros? Skip?


    # Age category - commented out in original code, keeping it commented
    # df['age_cat'] = df['age'].apply(categorize_age)
    # df.drop(columns=['age'], inplace=True)

    return df


def run_train(public_dir, model_dir, model_name='decision_tree'):
    os.makedirs(model_dir, exist_ok=True)

    # Load training data
    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    if not os.path.exists(train_path):
         print(f"Error: Training data file not found at {train_path}")
         sys.exit(1) # Exit if train file is missing

    df = pd.read_json(train_path, lines=True)

    # --- Check if dataframe is empty after loading ---
    if df.empty:
        print(f"Error: Training data loaded from {train_path} is empty.")
        sys.exit(1)

    df = engineer_features(df)

    # --- Check if dataframe is empty after engineering features ---
    if df.empty:
        print("Error: Training data became empty after feature engineering.")
        sys.exit(1)

    # Split features and label
    if 'two_year_recid' not in df.columns:
        print("Error: Target column 'two_year_recid' not found in the training data.")
        sys.exit(1)

    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    # Identify categorical and numerical columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Make sure 'priors_count' and 'race' are handled correctly by the pipeline
    priors_col = 'priors_count'
    race_col = 'race'

    # Check if 'priors_count' and 'race' are in the DataFrame before proceeding
    if priors_col not in X.columns:
         print(f"Error: Column '{priors_col}' required for NormalizeColumnByLabel not found in features.")
         sys.exit(1)
    if race_col not in X.columns:
         print(f"Error: Column '{race_col}' required for NormalizeColumnByLabel not found in features.")
         sys.exit(1)

    # Remove 'priors_count' from the standard numerical list as it's handled separately
    if priors_col in num_features:
        num_features.remove(priors_col)

    # Ensure 'race' is in categorical features if it exists
    if race_col not in cat_features:
         if race_col in X.columns: # Check if it's just not identified as object/category
             print(f"Warning: Column '{race_col}' is not identified as categorical/object. Adding to categorical features.")
             cat_features.append(race_col)
         # If it's not in columns at all, the earlier check would have exited.

    # Create lists of features to be passed to the ColumnTransformer
    # The 'remainder' will handle the output of NormalizeColumnByLabel which should replace the original 'priors_count'
    # Need to make sure the ColumnTransformer handles the columns correctly.
    # The simplest way is often to specify columns explicitly, but 'remainder'='passthrough' is used.
    # The NormalizeColumnByLabel step modifies 'priors_count' in place on a copy, then returns the modified copy.
    # This means the ColumnTransformer receives the DataFrame *with* the normalized 'priors_count'.

    # Add NormalizeColumnByLabel as the *first* step in the full pipeline
    # It's crucial that 'priors_count' and 'race' columns are present in the input DataFrame `X` at this stage.
    full_pipeline = Pipeline([
        # Step 1: Normalize 'priors_count' based on 'race'
        ('normalize_priors', NormalizeColumnByLabel(col=priors_col, label=race_col)),

        # Step 2: Apply other preprocessing steps using ColumnTransformer
        # This step now receives the DataFrame where 'priors_count' is potentially normalized.
        # It needs to handle numerical features (excluding original priors_count), and categorical features.
        ('preprocessing', ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='mean'), num_features),
                ('cat', Pipeline([
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ('imputer', SimpleImputer(strategy='most_frequent'))
                ]), cat_features) # cat_features should include 'race'
            ],
            # 'priors_count' is now part of the numerical features list again, but processed by step 1.
            # We need to make sure it's included in the output.
            # The 'remainder' handles columns not explicitly listed in transformers.
            # If 'priors_count' is NOT in num_features list given to 'num' transformer,
            # and NOT in cat_features list given to 'cat' transformer,
            # 'remainder'='passthrough' will include it.
            # Since we REMOVED 'priors_count' from num_features, it will be passed through.
            # This seems correct if NormalizeColumnByLabel is the first step and returns the whole modified DF.
            remainder='passthrough'
        ))
         # If StandardScaler/MinMaxScaler was needed for other numerical features, it would go here in the pipeline:
         # ('scaler', StandardScaler()), # or MinMaxScaler(), etc.
    ])


    # Fit preprocessing
    # --- This is where the error occurred ---
    try:
        X_processed = full_pipeline.fit_transform(X)
        print(f"Preprocessing successful. Processed data shape: {X_processed.shape}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        # Add debugging information: check X before fit_transform
        print(f"Shape of X before full_pipeline.fit_transform: {X.shape}")
        print(f"Columns in X: {X.columns.tolist()}")
        # Check unique race values and counts if race column exists
        if race_col in X.columns:
            print(f"Unique values in '{race_col}' and their counts:\n{X[race_col].value_counts(dropna=False)}")
            # Verify if any group has 0 counts
            for group in X[race_col].unique():
                 if len(X.loc[X[race_col] == group]) == 0:
                      print(f"Found 0 samples for group '{group}' in column '{race_col}'")
        else:
             print(f"Column '{race_col}' not found in X.")

        sys.exit(1) # Exit if preprocessing fails


    # Ensure X_processed is a DataFrame if needed for GridSearchCV or subsequent steps
    # ColumnTransformer with sparse_output=False and remainder='passthrough' usually returns a numpy array.
    # GridSearchCV and models typically work with numpy arrays, but it's good to be aware.
    # If you need column names or a DataFrame structure later, you might need to recreate it.
    # For now, assume numpy array is fine.

    # MODEL SELECTION
    # Ensure X_processed is not empty after preprocessing
    if X_processed.shape[0] == 0:
        print("Error: Processed data is empty after full pipeline transformation.")
        sys.exit(1)


    # The rest of your code for GridSearchCV remains the same...
    if model_name == 'decision_tree':
        base_model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': [None, 'sqrt', 'log2'],
            'max_leaf_nodes': [None, 10, 20, 50],
            'min_weight_fraction_leaf': [0.0, 0.01, 0.05],
            'class_weight': [None, 'balanced']
        }

    elif model_name == 'catboost':
        base_model = CatBoostClassifier(verbose=0, random_state=42)
        # Note: CatBoost can handle categorical features directly if specified.
        # The current pipeline one-hot encodes them first. This is fine, but less efficient for CatBoost.
        # If you wanted to use CatBoost's native categorical handling, you'd remove the OneHotEncoder
        # and pass the categorical feature indices to the CatBoost constructor.
        param_grid = {
            'iterations': [100, 200],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128],
            'scale_pos_weight': [1, 2, 5, 10],
            'bagging_temperature': [0, 0.5, 1, 2],
            'random_strength': [1, 5, 10],
            'rsm': [0.8, 1.0]  # Random subspace method (column sampling)
        }
    elif model_name == 'random_forest':
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced']
        }
    elif model_name == 'adaboost':
        # base_estimator param expects an unfitted estimator instance
        # The default base_estimator is DecisionTreeClassifier(max_depth=1)
        # If you want to tune the base_estimator, you need to pass it.
        # For example, base_model = AdaBoostClassifier(random_state=42)
        # Then param_grid can tune base_estimator__max_depth, etc.
        # Let's use the default base_estimator and tune only AdaBoost params first for simplicity
        base_model = AdaBoostClassifier(random_state=42) # Use default base_estimator
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
             # Example tuning base estimator:
             # 'base_estimator__max_depth': [1, 2, 3], # Requires passing base_estimator=DecisionTreeClassifier()
             # 'base_estimator__class_weight': [None, 'balanced']
        }
        # If you *do* want to tune base_estimator, you need to define it *before* the model.
        # E.g., base_tree = DecisionTreeClassifier(random_state=42) # can set default params here
        # base_model = AdaBoostClassifier(base_estimator=base_tree, random_state=42)
        # Then use the commented out param_grid for adaboost

    elif model_name == "xgboost":
        base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

        # Tìm kiếm tham số cho XGBoost
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'gamma': [0, 0.1, 0.5, 1],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [1, 2, 5, 10],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2],
            # 'booster': ['gbtree', 'gblinear'] # xgblinear doesn't support all other params, maybe tune separately or remove
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Grid search
    # Use X_processed (numpy array) here
    grid_search = GridSearchCV(base_model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1_macro', n_jobs=-1, verbose=2)
    print(f"Starting GridSearchCV for {model_name}...")
    try:
        grid_search.fit(X_processed, y)
        print("GridSearchCV finished.")
    except Exception as e:
        print(f"Error during GridSearchCV fit: {e}")
        # Add more context if needed
        sys.exit(1)


    best_model = grid_search.best_estimator_
    print(f"Best parameters for {model_name}:\n{grid_search.best_params_}\n")
    print("Training classification report (on training data with best model):")
    # Predict on X_processed (training data after preprocessing)
    y_pred_train = best_model.predict(X_processed)
    print(classification_report(y, y_pred_train))

    # Save model and preprocessor
    model_save_path = os.path.join(model_dir, 'trained_model.joblib')
    preprocessor_save_path = os.path.join(model_dir, 'preprocessor.joblib')

    try:
        dump(best_model, model_save_path)
        print(f"Best model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    try:
        dump(full_pipeline, preprocessor_save_path)
        print(f"Preprocessor saved to {preprocessor_save_path}")
    except Exception as e:
        print(f"Error saving preprocessor: {e}")



def run_predict(model_dir, test_input_dir, output_path):
    # Load model and preprocessor
    model_path = os.path.join(model_dir, 'trained_model.joblib')
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(preprocessor_path):
        print(f"Error: Preprocessor file not found at {preprocessor_path}")
        sys.exit(1)

    try:
        model = load(model_path)
        preprocessor = load(preprocessor_path)
        print("Model and preprocessor loaded successfully.")
    except Exception as e:
        print(f"Error loading model or preprocessor: {e}")
        sys.exit(1)


    # Load test data
    test_path = os.path.join(test_input_dir, 'test.json')
    if not os.path.exists(test_path):
         print(f"Error: Test data file not found at {test_path}")
         sys.exit(1)

    try:
        df_test = pd.read_json(test_path, lines=True)
        print(f"Test data loaded successfully. Shape: {df_test.shape}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)

    # --- Check if test dataframe is empty ---
    if df_test.empty:
        print(f"Warning: Test data loaded from {test_path} is empty. No predictions will be made.")
        pd.DataFrame({'two_year_recid': []}).to_json(output_path, orient='records', lines=True)
        print(f"Empty prediction file saved to {output_path}")
        return # Exit the function if test data is empty


    df_test = engineer_features(df_test)

    # --- Check if test dataframe is empty after engineering features ---
    if df_test.empty:
        print("Warning: Test data became empty after feature engineering. No predictions will be made.")
        pd.DataFrame({'two_year_recid': []}).to_json(output_path, orient='records', lines=True)
        print(f"Empty prediction file saved to {output_path}")
        return # Exit the function


    # Transform and predict
    print("Transforming test data...")
    try:
        X_test = preprocessor.transform(df_test)
        print(f"Test data transformed successfully. Processed shape: {X_test.shape}")
    except Exception as e:
        print(f"Error during test data transformation: {e}")
        # Add debugging context
        print(f"Shape of df_test before preprocessor.transform: {df_test.shape}")
        print(f"Columns in df_test: {df_test.columns.tolist()}")
        # Check unique race values and counts if race column exists
        race_col = 'race'
        if race_col in df_test.columns:
            print(f"Unique values in '{race_col}' and their counts (test data):\n{df_test[race_col].value_counts(dropna=False)}")
        sys.exit(1) # Exit if transformation fails

    # Ensure X_test is not empty after preprocessing
    if X_test.shape[0] == 0:
        print("Warning: Processed test data is empty after transformation. No predictions will be made.")
        pd.DataFrame({'two_year_recid': []}).to_json(output_path, orient='records', lines=True)
        print(f"Empty prediction file saved to {output_path}")
        return


    print("Making predictions...")
    try:
        preds = model.predict(X_test)
        print("Predictions made successfully.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

    # Save predictions
    try:
        pd.DataFrame({'two_year_recid': preds}).to_json(output_path, orient='records', lines=True)
        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Train command
    parser_train = subparsers.add_parser('train', help='Tune models using Grid Search, create best voting ensemble, save preprocessor and model')
    parser_train.add_argument('--public_dir', type=str, default='./', help='Directory containing train_data/train.json')
    parser_train.add_argument('--model_dir', type=str, default ='./', help='Directory to save preprocessor.joblib and trained_voting_model.joblib')
    parser_train.add_argument('--model_name', type=str, default='decision_tree', choices=['decision_tree', 'catboost', 'random_forest', 'adaboost', 'xgboost'], help='Model to train')

    # Predict command
    parser_predict = subparsers.add_parser('predict', help='Make predictions using saved preprocessor and voting model')
    parser_predict.add_argument('--model_dir', type=str, default='./', help='Directory containing preprocessor.joblib and trained_..._voting_model.joblib')
    parser_predict.add_argument('--test_input_dir', type=str, default='./test_data', help='Directory containing test.json')
    parser_predict.add_argument('--output_path', type=str, default='./predict.json', help='File path for saving predictions')

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args.public_dir, args.model_dir, model_name=args.model_name)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.test_input_dir, args.output_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()