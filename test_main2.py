import argparse
import sys
import os
import pandas as pd
import numpy as np
import optuna
import warnings

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


from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from joblib import dump, load
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        # Added checks to ensure quantiles are within valid range [0, 1] and lower <= upper
        if not 0 <= lower_quantile <= 1:
            raise ValueError("lower_quantile must be between 0 and 1")
        if not 0 <= upper_quantile <= 1:
            raise ValueError("upper_quantile must be between 0 and 1")
        if lower_quantile > upper_quantile:
             # Handle potential issue where suggestion might make lower > upper
             # A simple approach is to swap them or raise an error.
             # Let's swap them and print a warning, or rely on Optuna to learn valid ranges.
             # For simplicity with Optuna, suggesting ranges that make this unlikely is best.
             # We'll add a check, but primarily rely on suggest_float ranges.
             if lower_quantile > upper_quantile:
                  warnings.warn(f"lower_quantile ({lower_quantile}) > upper_quantile ({upper_quantile}). Swapping them.", UserWarning)
                  self.lower_quantile = upper_quantile
                  self.upper_quantile = lower_quantile
             else:
                  self.lower_quantile = lower_quantile
                  self.upper_quantile = upper_quantile
        else:
            self.lower_quantile = lower_quantile
            self.upper_quantile = upper_quantile


    def fit(self, X, y=None):
        # Ensure X is numpy array or pandas DataFrame for quantile calculation
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        elif isinstance(X, np.ndarray):
            X_np = X
        else:
             raise TypeError("Input X must be pandas DataFrame or numpy array")

        # Handle potential NaN values if Winsorizer is before imputation (though usually after)
        # Quantile calculation in numpy/pandas usually handles NaNs by default, but let's be explicit if needed
        # Given our pipeline structure, imputation is before Winsorizer, so NaNs shouldn't be an issue here.

        self.lower_bounds_ = np.nanquantile(X_np, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X_np, self.upper_quantile, axis=0)

        # Handle cases where all values in a column are the same (quantile might be NaN or constant)
        # If lower_bounds_ == upper_bounds_ for a column, clipping won't do anything.
        # np.clip handles lower >= upper case gracefully (results in the upper/lower value itself)

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        elif isinstance(X, np.ndarray):
            X_np = X
        else:
             raise TypeError("Input X must be pandas DataFrame or numpy array")

        # Apply clipping using fitted bounds
        X_clipped = np.clip(X_np, self.lower_bounds_, self.upper_bounds_)
        return X_clipped

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
    # Age category
    # Check if 'age' column exists before dropping
    if 'age' in df.columns:
        df['age_cat'] = df['age'].apply(categorize_age)
        df.drop(columns=['age'], inplace=True)
    else:
        print("Warning: 'age' column not found for feature engineering.")


    # Ensure columns exist before accessing
    juv_fel_count = df['juv_fel_count'] if 'juv_fel_count' in df.columns else 0
    juv_misd_count = df['juv_misd_count'] if 'juv_misd_count' in df.columns else 0
    juv_other_count = df['juv_other_count'] if 'juv_other_count' in df.columns else 0
    priors_count = df['priors_count'] if 'priors_count' in df.columns else 0
    c_charge_degree = df['c_charge_degree'] if 'c_charge_degree' in df.columns else pd.Series([''] * len(df), index=df.index) # Default to empty strings or similar if missing


    # Tổng số vi phạm vị thành niên
    df['juv_total_count'] = juv_fel_count + juv_misd_count + juv_other_count

    # Có từng phạm pháp khi vị thành niên không?
    # Ensure sum is treated as numeric, fillna(0) if necessary before > 0 comparison
    df['is_juvenile_offender'] = (df['juv_total_count'].fillna(0) > 0).astype(int)

    # Đã từng phạm pháp trước đó?
    df['is_repeat_offender'] = (priors_count.fillna(0) > 0).astype(int)

    # Có phạm tội nghiêm trọng không?
    # Use .astype(int) carefully after comparison
    df['is_felony'] = (c_charge_degree == 'F').astype(int)


    return df

def get_model_and_params(trial, model_type):
    """Suggest hyperparameters for a given model type using Optuna trial."""
    model = None
    params = {} # Store model-specific parameters here

    if model_type == 'knn':
        params['knn_n_neighbors'] = trial.suggest_int('knn_n_neighbors', 3, 15)
        params['knn_weights'] = trial.suggest_categorical('knn_weights', ['uniform', 'distance'])
        params['knn_metric'] = trial.suggest_categorical('knn_metric', ['minkowski', 'manhattan'])
        model = KNeighborsClassifier(n_neighbors=params['knn_n_neighbors'], weights=params['knn_weights'], metric=params['knn_metric'], n_jobs=-1)

    elif model_type == 'logistic':
        params['logreg_c'] = trial.suggest_float("logreg_c", 1e-2, 1e1, log=True)
        params['logreg_penalty'] = trial.suggest_categorical('logreg_penalty', ['l2'])
        solver = 'liblinear' # Keep liblinear as it's simple and good for binary L1/L2
        model = LogisticRegression(C=params['logreg_c'], penalty=params['logreg_penalty'], solver=solver, max_iter=2000, random_state=42, n_jobs=-1)

    elif model_type == 'svc':
        params['svc_c'] = trial.suggest_float("svc_c", 1e-1, 1e2, log=True)
        params['svc_kernel'] = trial.suggest_categorical("svc_kernel", ["linear", "rbf"])
        gamma = "scale"
        degree = 3

        if params['svc_kernel'] == "rbf":
            params['svc_gamma_type'] = trial.suggest_categorical("svc_gamma_type", ["scale", "float"])
            if params['svc_gamma_type'] == "float":
                params['svc_gamma_float'] = trial.suggest_float("svc_gamma_float", 1e-3, 1.0, log=True)
                gamma = params['svc_gamma_float']
            else:
                gamma = params['svc_gamma_type']

        params['_svc_calculated_gamma'] = gamma
        params['_svc_calculated_degree'] = degree

        model = SVC(C=params['svc_c'], kernel=params['svc_kernel'], gamma=gamma, degree=degree, probability=True, random_state=42)

    elif model_type == 'decision_tree':
        params['dt_criterion'] = trial.suggest_categorical("dt_criterion", ["gini", "entropy"])
        params['dt_max_depth'] = trial.suggest_int("dt_max_depth", 5, 25)
        params['dt_min_samples_split'] = trial.suggest_int("dt_min_samples_split", 2, 10)
        params['dt_min_samples_leaf'] = trial.suggest_int("dt_min_samples_leaf", 1, 10)
        model = DecisionTreeClassifier(
            criterion=params['dt_criterion'],
            max_depth=params['dt_max_depth'],
            min_samples_split=params['dt_min_samples_split'],
            min_samples_leaf=params['dt_min_samples_leaf'],
            random_state=42
        )

    elif model_type == 'random_forest':
        params['rf_n_estimators'] = trial.suggest_int("rf_n_estimators", 50, 200)
        params['rf_max_depth_type'] = trial.suggest_categorical("rf_max_depth_type", ["int", "none"])
        max_depth = None
        if params['rf_max_depth_type'] == "int":
             params['rf_max_depth_int'] = trial.suggest_int("rf_max_depth_int", 10, 30)
             max_depth = params['rf_max_depth_int']
        else:
             max_depth = None

        params['rf_min_samples_split'] = trial.suggest_int("rf_min_samples_split", 2, 10)
        params['rf_min_samples_leaf'] = trial.suggest_int("rf_min_samples_leaf", 1, 10)
        params['rf_max_features_type'] = trial.suggest_categorical("rf_max_features_type", ["sqrt", "log2"])
        max_features = params['rf_max_features_type']

        params['rf_criterion'] = trial.suggest_categorical("rf_criterion", ["gini", "entropy"])
        params['_rf_calculated_max_depth'] = max_depth
        params['_rf_calculated_max_features'] = max_features

        model = RandomForestClassifier(
            n_estimators=params['rf_n_estimators'],
            max_depth=max_depth,
            min_samples_split=params['rf_min_samples_split'],
            min_samples_leaf=params['rf_min_samples_leaf'],
            max_features=max_features,
            criterion=params['rf_criterion'],
            random_state=42,
            n_jobs=-1
        )

    elif model_type == 'gradient_boosting':
        params['gb_n_estimators'] = trial.suggest_int("gb_n_estimators", 50, 200)
        params['gb_learning_rate'] = trial.suggest_float("gb_learning_rate", 1e-2, 0.3, log=True)
        params['gb_max_depth'] = trial.suggest_int("gb_max_depth", 3, 7)
        params['gb_subsample'] = trial.suggest_float("gb_subsample", 0.7, 1.0)
        params['gb_min_samples_split'] = trial.suggest_int("gb_min_samples_split", 2, 10)
        params['gb_min_samples_leaf'] = trial.suggest_int("gb_min_samples_leaf", 1, 10)
        model = GradientBoostingClassifier(
            n_estimators=params['gb_n_estimators'],
            learning_rate=params['gb_learning_rate'],
            max_depth=params['gb_max_depth'],
            subsample=params['gb_subsample'],
            min_samples_split=params['gb_min_samples_split'],
            min_samples_leaf=params['gb_min_samples_leaf'],
            random_state=42
        )

    elif model_type == 'gaussian_nb':
        params['gnb_var_smoothing'] = trial.suggest_float("gnb_var_smoothing", 1e-9, 1e-5, log=True)
        model = GaussianNB(var_smoothing=params['gnb_var_smoothing'])

    elif model_type == 'mlp':
        num_hidden_layers = trial.suggest_int("mlp_num_hidden_layers", 1, 2)
        layer_sizes = []
        for i in range(num_hidden_layers):
            layer_size = trial.suggest_int(f"mlp_n_neurons_layer_{i}", 50, 150)
            layer_sizes.append(layer_size)
        params['mlp_hidden_layer_sizes'] = tuple(layer_sizes) # Store as tuple in params

        params['mlp_activation'] = trial.suggest_categorical("mlp_activation", ["relu", "tanh"])
        params['mlp_alpha'] = trial.suggest_float("mlp_alpha", 1e-5, 1e-3, log=True)
        params['mlp_learning_rate_init'] = trial.suggest_float("mlp_learning_rate_init", 1e-3, 1e-2, log=True)
        model = MLPClassifier(
            hidden_layer_sizes=params['mlp_hidden_layer_sizes'], # Use the tuple from params
            activation=params['mlp_activation'],
            solver='adam',
            alpha=params['mlp_alpha'],
            learning_rate_init=params['mlp_learning_rate_init'],
            max_iter=1500,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42
        )

    elif model_type == 'xgboost':
        params['xgb_n_estimators'] = trial.suggest_int("xgb_n_estimators", 100, 300)
        params['xgb_learning_rate'] = trial.suggest_float("xgb_learning_rate", 1e-2, 0.2, log=True)
        params['xgb_max_depth'] = trial.suggest_int("xgb_max_depth", 3, 7)
        params['xgb_subsample'] = trial.suggest_float("xgb_subsample", 0.6, 0.9)
        params['xgb_colsample_bytree'] = trial.suggest_float("xgb_colsample_bytree", 0.6, 0.9)
        params['xgb_gamma'] = trial.suggest_float("xgb_gamma", 0, 0.5)
        params['xgb_reg_alpha'] = trial.suggest_float('xgb_reg_alpha', 1e-2, 10.0, log=True)
        params['xgb_reg_lambda'] = trial.suggest_float('xgb_reg_lambda', 1e-2, 10.0, log=True)

        model = xgb.XGBClassifier(
            n_estimators=params['xgb_n_estimators'],
            learning_rate=params['xgb_learning_rate'],
            max_depth=params['xgb_max_depth'],
            subsample=params['xgb_subsample'],
            colsample_bytree=params['xgb_colsample_bytree'],
            gamma=params['xgb_gamma'],
            reg_alpha=params['xgb_reg_alpha'],
            reg_lambda=params['xgb_reg_lambda'],
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

    elif model_type == 'lightgbm':
        params['lgbm_n_estimators'] = trial.suggest_int("lgbm_n_estimators", 100, 300)
        params['lgbm_learning_rate'] = trial.suggest_float("lgbm_learning_rate", 1e-2, 0.2, log=True)
        params['lgbm_num_leaves'] = trial.suggest_int("lgbm_num_leaves", 20, 50)
        params['lgbm_max_depth'] = trial.suggest_int("lgbm_max_depth", 3, 7)
        params['lgbm_subsample'] = trial.suggest_float("lgbm_subsample", 0.6, 0.9)
        params['lgbm_colsample_bytree'] = trial.suggest_float("lgbm_colsample_bytree", 0.6, 0.9)
        params['lgbm_reg_alpha'] = trial.suggest_float('lgbm_reg_alpha', 1e-2, 10.0, log=True)
        params['lgbm_reg_lambda'] = trial.suggest_float('lgbm_reg_lambda', 1e-2, 10.0, log=True)

        model = lgb.LGBMClassifier(
            n_estimators=params['lgbm_n_estimators'],
            learning_rate=params['lgbm_learning_rate'],
            num_leaves=params['lgbm_num_leaves'],
            max_depth=params['lgbm_max_depth'],
            subsample=params['lgbm_subsample'],
            colsample_bytree=params['lgbm_colsample_bytree'],
            reg_alpha=params['lgbm_reg_alpha'],
            reg_lambda=params['lgbm_reg_lambda'],
            random_state=42,
            n_jobs=-1,
            verbose = -1
        )

    elif model_type == 'catboost':
        params['cb_iterations'] = trial.suggest_int("cb_iterations", 100, 300)
        params['cb_learning_rate'] = trial.suggest_float("cb_learning_rate", 1e-2, 0.2, log=True)
        params['cb_depth'] = trial.suggest_int("cb_depth", 3, 7)
        params['cb_l2_leaf_reg'] = trial.suggest_float("cb_l2_leaf_reg", 1, 10, log=True)
        params['cb_border_count'] = trial.suggest_int("cb_border_count", 32, 128)
        params['cb_random_strength'] = trial.suggest_float('cb_random_strength', 0.1, 2.0, log=True)

        model = CatBoostClassifier(
            iterations=params['cb_iterations'],
            learning_rate=params['cb_learning_rate'],
            depth=params['cb_depth'],
            l2_leaf_reg=params['cb_l2_leaf_reg'],
            border_count=params['cb_border_count'],
            random_strength=params['cb_random_strength'],
            random_state=42,
            verbose=0,
            thread_count=-1
        )


    elif model_type == 'adaboost':
        params['ada_base_max_depth'] = trial.suggest_int("ada_base_max_depth", 1, 4)
        base_estimator = DecisionTreeClassifier(max_depth=params['ada_base_max_depth'], random_state=42)

        params['ada_n_estimators'] = trial.suggest_int("ada_n_estimators", 50, 200)
        params['ada_learning_rate'] = trial.suggest_float("ada_learning_rate", 0.01, 1.0, log=True)

        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=params['ada_n_estimators'],
            learning_rate=params['ada_learning_rate'],
            random_state=42
        )

    return model, params # Return the model instance and the dictionary of model-specific parameters


def objective_function(trial, X_train, y_train, X_val, y_val, num_features, cat_features, model_type):
    """Optuna objective function to train and evaluate a pipeline."""

    # --- Preprocessing Hyperparameters ---
    # Imputation strategy for numerical features
    num_imputer_strategy = trial.suggest_categorical('num_imputer_strategy', ['mean', 'median', 'knn'])
    if num_imputer_strategy == 'knn':
        knn_imputer_n_neighbors = trial.suggest_int('knn_imputer_n_neighbors', 5, 20)
    else:
        knn_imputer_n_neighbors = None

    # Imputation strategy for categorical features
    cat_imputer_strategy = trial.suggest_categorical('cat_imputer_strategy', ['most_frequent', 'constant'])
    cat_imputer_fill_value = 'missing' if cat_imputer_strategy == 'constant' else None

    # Scaling type
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'robust'])
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else: # robust
        scaler = RobustScaler()

    # Log transformation for numerical features
    apply_log_transform = trial.suggest_categorical('apply_log_transform', [True, False])

    # Winsorizer quantiles - NOW TUNABLE
    # Suggest ranges that make lower < upper highly probable
    winsorizer_lower_quantile = trial.suggest_float('winsorizer_lower_quantile', 0.001, 0.1) # e.g., 0.1% to 10%
    winsorizer_upper_quantile = trial.suggest_float('winsorizer_upper_quantile', 0.9, 0.999) # e.g., 90% to 99.9%
    # Add a check just in case, though suggest_float ranges usually suffice
    if winsorizer_lower_quantile >= winsorizer_upper_quantile:
         # Simple strategy: if bounds cross, use default or reverse them.
         # Reversing might explore an inverted space, using defaults is safer.
         # Or just let the trial potentially fail or result in non-informative clipping.
         # Let's stick to the suggested values as np.clip handles order gracefully.
         pass # np.clip handles cases where a < b or a >= b

    # One-Hot Encoding parameters
    ohe_handle_unknown = trial.suggest_categorical('ohe_handle_unknown', ['ignore', 'infrequent_if_exist'])
    ohe_min_frequency = trial.suggest_float('ohe_min_frequency', 0.005, 0.05) if ohe_handle_unknown == 'infrequent_if_exist' else None


    # --- Build Preprocessing Pipelines ---
    numeric_steps = []
    # 1. Imputation
    if num_imputer_strategy == 'knn':
        numeric_steps.append(('imputer', KNNImputer(n_neighbors=knn_imputer_n_neighbors)))
    else:
        numeric_steps.append(('imputer', SimpleImputer(strategy=num_imputer_strategy)))

    # 2. Winsorization - Use the tuned parameters
    numeric_steps.append(('winsorizer', Winsorizer(lower_quantile=winsorizer_lower_quantile, upper_quantile=winsorizer_upper_quantile)))

    # 3. Log Transformation (Apply after imputation and winsorization)
    if apply_log_transform:
        numeric_steps.append(('log_transform', FunctionTransformer(func=np.log1p, validate=False)))

    # 4. Scaling (Apply after imputation, winsorization, and transformation)
    numeric_steps.append(('scaler', scaler))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cat_imputer_strategy, fill_value=cat_imputer_fill_value)),
        ('onehot', OneHotEncoder(handle_unknown=ohe_handle_unknown, min_frequency=ohe_min_frequency, sparse_output=False))
    ])

    trial_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough',
        n_jobs=-1
    )

    # --- Model and its Hyperparameters ---
    model, model_params = get_model_and_params(trial, model_type)

    # --- Full Pipeline and Evaluation ---
    full_pipeline = Pipeline(steps=[('preprocessor', trial_preprocessor), ('classifier', model)])

    # Fit and evaluate
    try:
        full_pipeline.fit(X_train, y_train)
        y_pred_val = full_pipeline.predict(X_val)
        score = f1_score(y_val, y_pred_val, average='macro')

        # Report score to Optuna
        trial.report(score, step=0)

        # Prune trial if needed
        if trial.should_prune():
            raise optuna.TrialPruned()

    except Exception as e:
        # Report a low score on failure
        print(f"Trial {trial.number} failed for {model_type} with error: {e}") # Print error for debugging
        score = -1.0
        # Check if trial is valid before reporting
        if trial and not trial.state.is_terminal():
            trial.report(score, step=0)
        else:
             # If trial is already pruned/failed, just return low score
             pass # Optuna might handle terminal states internally


    return score

def reconstruct_model(model_type, model_params):
    """Reconstruct a model instance from its type and parameters dictionary."""
    model = None
    if model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=model_params.get('knn_n_neighbors', 5), # Use .get with defaults for robustness
            weights=model_params.get('knn_weights', 'uniform'),
            metric=model_params.get('knn_metric', 'minkowski'),
            n_jobs=-1
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=model_params.get('logreg_c', 1.0),
            penalty=model_params.get('logreg_penalty', 'l2'),
            solver='liblinear',
            max_iter=2000,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'svc':
         gamma_val = model_params.get('_svc_calculated_gamma', 'scale')
         degree_val = model_params.get('_svc_calculated_degree', 3)
         model = SVC(
            C=model_params.get('svc_c', 1.0),
            kernel=model_params.get('svc_kernel', 'rbf'),
            gamma=gamma_val,
            degree=degree_val,
            probability=True,
            random_state=42
        )
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            criterion=model_params.get('dt_criterion', 'gini'),
            max_depth=model_params.get('dt_max_depth', 10),
            min_samples_split=model_params.get('dt_min_samples_split', 2),
            min_samples_leaf=model_params.get('dt_min_samples_leaf', 1),
            random_state=42
        )
    elif model_type == 'random_forest':
         # Retrieve calculated values first, fallback to suggesting parameters if not found (shouldn't happen if params are stored correctly)
         max_depth_val = model_params.get('_rf_calculated_max_depth') # This stores None or the int
         max_features_val = model_params.get('_rf_calculated_max_features') # This stores "sqrt" or "log2"

         model = RandomForestClassifier(
            n_estimators=model_params.get('rf_n_estimators', 100),
            max_depth=max_depth_val, # Use the calculated/stored value
            min_samples_split=model_params.get('rf_min_samples_split', 2),
            min_samples_leaf=model_params.get('rf_min_samples_leaf', 1),
            max_features=max_features_val, # Use the calculated/stored value
            criterion=model_params.get('rf_criterion', 'gini'),
            random_state=49, # Use a different random state for the final ensemble models if desired, or keep 42
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=model_params.get('gb_n_estimators', 100),
            learning_rate=model_params.get('gb_learning_rate', 0.1),
            max_depth=model_params.get('gb_max_depth', 3),
            subsample=model_params.get('gb_subsample', 1.0),
            min_samples_split=model_params.get('gb_min_samples_split', 2),
            min_samples_leaf=model_params.get('gb_min_samples_leaf', 1),
            random_state=42
        )
    elif model_type == 'gaussian_nb':
        model = GaussianNB(
            var_smoothing=model_params.get('gnb_var_smoothing', 1e-9)
        )
    elif model_type == 'mlp':
        # Reconstruct the hidden_layer_sizes tuple
        # Need to handle cases where num_hidden_layers might not be in stored params if tuning failed early?
        # Assuming successful tuning, num_hidden_layers and layer sizes should be there.
        num_hidden_layers = model_params.get('mlp_num_hidden_layers', 1) # Default to 1 layer
        layer_sizes = []
        for i in range(num_hidden_layers):
             # Safely get layer size, default to a reasonable size if missing
             layer_sizes.append(model_params.get(f"mlp_n_neurons_layer_{i}", 100))
        hidden_layer_sizes_tuple = tuple(layer_sizes)


        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes_tuple,
            activation=model_params.get('mlp_activation', 'relu'),
            solver='adam', # Fixed solver
            alpha=model_params.get('mlp_alpha', 0.0001),
            learning_rate_init=model_params.get('mlp_learning_rate_init', 0.001),
            max_iter=1500,
            early_stopping=False, # Turn off early stopping for final training
            random_state=42
        )
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=model_params.get('xgb_n_estimators', 100),
            learning_rate=model_params.get('xgb_learning_rate', 0.1),
            max_depth=model_params.get('xgb_max_depth', 6),
            subsample=model_params.get('xgb_subsample', 1.0),
            colsample_bytree=model_params.get('xgb_colsample_bytree', 1.0),
            gamma=model_params.get('xgb_gamma', 0),
            reg_alpha=model_params.get('xgb_reg_alpha', 0),
            reg_lambda=model_params.get('xgb_reg_lambda', 1),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators=model_params.get('lgbm_n_estimators', 100),
            learning_rate=model_params.get('lgbm_learning_rate', 0.1),
            num_leaves=model_params.get('lgbm_num_leaves', 31),
            max_depth=model_params.get('lgbm_max_depth', -1), # -1 means no limit
            subsample=model_params.get('lgbm_subsample', 1.0),
            colsample_bytree=model_params.get('lgbm_colsample_bytree', 1.0),
            reg_alpha=model_params.get('lgbm_reg_alpha', 0),
            reg_lambda=model_params.get('lgbm_reg_lambda', 0),
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif model_type == 'catboost':
        model = CatBoostClassifier(
            iterations=model_params.get('cb_iterations', 100),
            learning_rate=model_params.get('cb_learning_rate', 0.1),
            depth=model_params.get('cb_depth', 6),
            l2_leaf_reg=model_params.get('cb_l2_leaf_reg', 3),
            border_count=model_params.get('cb_border_count', 254),
            random_strength=model_params.get('cb_random_strength', 1.0),
            random_state=42,
            verbose=0,
            thread_count=-1
        )
    elif model_type == 'adaboost':
        # Reconstruct the base estimator first
        base_estimator_max_depth = model_params.get('ada_base_max_depth', 1)
        base_estimator = DecisionTreeClassifier(max_depth=base_estimator_max_depth, random_state=42)

        model = AdaBoostClassifier(
            estimator=base_estimator, # Use the reconstructed base estimator
            n_estimators=model_params.get('ada_n_estimators', 50),
            learning_rate=model_params.get('ada_learning_rate', 1.0),
            random_state=42
        )


    if model is None:
        raise ValueError(f"Failed to reconstruct model type '{model_type}'")
    return model

def build_final_preprocessor(preprocessor_params, num_features, cat_features):
    """Build the final preprocessor pipeline using the best parameters from Optuna."""

    # Extract all preprocessor parameters, including Winsorizer quantiles
    num_imputer_strategy = preprocessor_params.get('num_imputer_strategy', 'mean')
    knn_imputer_n_neighbors = preprocessor_params.get('knn_imputer_n_neighbors', 5) # Default for KNN
    apply_log_transform = preprocessor_params.get('apply_log_transform', False)

    # Get TUNED Winsorizer parameters, default to 0.01, 0.99 if not found
    winsorizer_lower_quantile = preprocessor_params.get('winsorizer_lower_quantile', 0.01)
    winsorizer_upper_quantile = preprocessor_params.get('winsorizer_upper_quantile', 0.99)


    scaler_type = preprocessor_params.get('scaler_type', 'standard')
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else: # robust
        scaler = RobustScaler()

    cat_imputer_strategy = preprocessor_params.get('cat_imputer_strategy', 'most_frequent')
    cat_imputer_fill_value = 'missing' if cat_imputer_strategy == 'constant' else None

    ohe_handle_unknown = preprocessor_params.get('ohe_handle_unknown', 'ignore')
    # Default for min_frequency only needed if handle_unknown was 'infrequent_if_exist'
    # Let's use a default that won't break 'ignore' and is reasonable if applied
    ohe_min_frequency = preprocessor_params.get('ohe_min_frequency', None) # None is correct default for ignore


    # Build numeric pipeline steps using the best parameters
    numeric_steps = []
    # 1. Imputation
    if num_imputer_strategy == 'knn':
        # Ensure knn_imputer_n_neighbors is used if strategy is 'knn'
        numeric_steps.append(('imputer', KNNImputer(n_neighbors=knn_imputer_n_neighbors)))
    else:
        numeric_steps.append(('imputer', SimpleImputer(strategy=num_imputer_strategy)))

    # 2. Winsorization - Use the TUNED parameters
    numeric_steps.append(('winsorizer', Winsorizer(lower_quantile=winsorizer_lower_quantile, upper_quantile=winsorizer_upper_quantile)))

    # 3. Log Transformation (Apply after imputation and winsorization)
    if apply_log_transform:
         numeric_steps.append(('log_transform', FunctionTransformer(func=np.log1p, validate=False)))

    # 4. Scaling (Apply after imputation, winsorization, and transformation)
    numeric_steps.append(('scaler', scaler))

    numeric_transformer = Pipeline(steps=numeric_steps)


    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cat_imputer_strategy, fill_value=cat_imputer_fill_value)),
        ('onehot', OneHotEncoder(handle_unknown=ohe_handle_unknown, min_frequency=ohe_min_frequency, sparse_output=False))
    ])

    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough',
        n_jobs=-1
    )
    return final_preprocessor


def run_train(public_dir, model_dir, n_trials_per_model=30):
    print("--- Starting Training ---")
    os.makedirs(model_dir, exist_ok=True)

    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    if not os.path.exists(train_path):
        print(f"Error: Training data file not found at {train_path}. Exiting.")
        sys.exit(1)

    df = pd.read_json(train_path, lines=True)
    print(f"Loaded {len(df)} training cases.")

    if df.empty:
        print("Error: Training data is empty. Exiting.")
        sys.exit(1)


    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    # Drop non-predictive or potentially problematic columns based on typical data
    # These columns might be leaky or non-existent in new data
    # Added removal of id, name, first, last, compas_screening_date, dob, days_b_screening_arrest, c_jail_in, c_jail_out, c_offense_date, c_arrest_date, screening_date, violent_recid, is_violent_recid
    # Keep 'race' for now as it's often used, but be mindful of bias implications.
    # Removed c_case_number and decile_score as they are likely derived/metadata.
    cols_to_drop = [
        'id', 'name', 'first', 'last', 'compas_screening_date', 'dob',
        'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_offense_date',
        'c_arrest_date', 'screening_date', 'violent_recid', 'is_violent_recid',
        'c_case_number', 'decile_score'
    ]
    # Drop only columns that actually exist in the dataframe
    cols_to_drop_existing = [col for col in cols_to_drop if col in X.columns]
    if cols_to_drop_existing:
        X = X.drop(columns=cols_to_drop_existing)
        print(f"Dropped non-predictive columns: {cols_to_drop_existing}")


    X = engineer_features(X)

    # Re-determine features after engineering and dropping columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure target variable has at least 2 classes for stratification
    # Check unique values count and distribution
    if len(y.unique()) < 2:
         print("Error: Target variable does not have at least 2 classes after data loading. Cannot stratify or train. Exiting.")
         sys.exit(1)

    # Check for minimum samples per class for stratification
    min_samples_per_class = y.value_counts().min()
    if min_samples_per_class < 2:
         print(f"Warning: Minimum samples per class is {min_samples_per_class}. Stratification may not be possible or reliable.")
         stratify_option = None # Disable stratification if not enough samples in any class
    else:
         stratify_option = y


    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option # Use stratify_option
    )
    print(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}")


    supported_models = [
        'knn', 'logistic', 'svc', 'decision_tree', 'random_forest',
        #'gradient_boosting', 'gaussian_nb', 'mlp', # Commented out less performant/slower models for speed
        'xgboost', 'lightgbm', 'catboost', 'adaboost'
    ]
    all_model_results = {}
    studies = {} # Store study objects to potentially access best trial parameters

    print(f"\n--- Starting Optuna Hyperparameter Tuning ({n_trials_per_model} Trials Per Model) ---")

    for model_type in tqdm(supported_models, desc="Model Types Tuned"):
        print(f"\nTuning {model_type.upper()}...")

        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

        objective_func = lambda trial: objective_function(trial, X_train, y_train, X_val, y_val, num_features, cat_features, model_type)

        try:
            study.optimize(
                objective_func,
                n_trials=n_trials_per_model,
                n_jobs=-1,
                show_progress_bar=False
            )

            if study.best_trial and study.best_trial.value > -1.0: # Check if a valid trial completed
                all_model_results[model_type] = {
                    'best_score': study.best_value,
                    'best_params': study.best_trial.params
                }
                studies[model_type] = study
                print(f"  Best F1 Macro for {model_type.upper()}: {study.best_value:.4f}")
                # print(f"  Best Params: {study.best_trial.params}") # Optional: print all best params per model
            else:
                 print(f"  No valid trials completed successfully for {model_type.upper()} (score <= -1.0).")


        except Exception as e:
            print(f"An unhandled error occurred during tuning for {model_type.upper()}: {e}")
            # Continue to the next model type


    print(f"\n--- Optuna Tuning Finished ---")

    # Filter out models that didn't yield valid results
    valid_model_results = {k: v for k, v in all_model_results.items() if v['best_score'] > -1.0}

    if not valid_model_results:
        print("Error: Optuna tuning failed or produced no valid results for all tested model types. Cannot ensemble. Exiting.")
        sys.exit(1)

    # Sort models by their best validation score
    sorted_models = sorted(valid_model_results.items(), key=lambda item: item[1]['best_score'], reverse=True)

    # Determine how many top models to ensemble
    top_n = 3
    if len(sorted_models) < top_n:
        print(f"Warning: Fewer than {top_n} models had valid tuning results ({len(sorted_models)}). Using all available valid models.")
        top_n = len(sorted_models)

    if top_n == 0:
        print("Error: No valid models available to form an ensemble after tuning. Exiting.")
        sys.exit(1)


    top_models_info = sorted_models[:top_n]

    print("\n--- Top Performing Models for Ensemble ---")
    for i, (model_type, results) in enumerate(top_models_info):
        print(f"{i+1}. Model: {model_type.upper()}, Best Val F1 Macro: {results['best_score']:.4f}")
        # print(f"   Best params sample: {list(results['best_params'].items())[:5]}...") # Show sample params


    # Use the preprocessor parameters from the single best trial among *all* tuned models
    # This is the best trial across all model studies
    overall_best_trial_across_models = None
    overall_best_score = -float('inf')

    for model_type, study in studies.items():
        if study and study.best_trial and study.best_trial.value > overall_best_score:
             overall_best_score = study.best_trial.value
             overall_best_trial_across_models = study.best_trial

    if overall_best_trial_across_models is None:
        print("Error: Could not identify a single best trial across all studies. Cannot build preprocessor. Exiting.")
        sys.exit(1)


    print(f"\nUsing preprocessor settings from the overall best trial (score: {overall_best_score:.4f}).")
    # Extract all preprocessor-specific parameters from this single best trial
    # Updated list to include winsorizer parameters
    preprocessor_param_keys = [
        'num_imputer_strategy', 'knn_imputer_n_neighbors', 'apply_log_transform',
        'winsorizer_lower_quantile', 'winsorizer_upper_quantile', # Added winsorizer keys
        'scaler_type', 'cat_imputer_strategy', 'ohe_handle_unknown', 'ohe_min_frequency'
    ]
    # Extract these keys from the overall best trial's full parameter dictionary
    best_preprocessor_params = {k: overall_best_trial_across_models.params.get(k, None) for k in preprocessor_param_keys}
    # Remove None values if a parameter wasn't applicable or suggested in that trial
    best_preprocessor_params = {k: v for k, v in best_preprocessor_params.items() if v is not None}


    print("Overall Best Trial's Preprocessor Parameters Selected:")
    for key, value in best_preprocessor_params.items():
        print(f"  {key}: {value}")


    # Build the final preprocessor using the selected parameters from the overall best trial
    final_preprocessor = build_final_preprocessor(best_preprocessor_params, num_features, cat_features)


    # Reconstruct the top models using their respective best tuning parameters for the ensemble
    estimators = []
    print("\nReconstructing Top Models for Ensemble:")
    for model_type, results in top_models_info:
        # Get the best trial parameters for *this specific model type*
        model_specific_best_params = results['best_params']

        # Extract *only* model specific params, excluding preprocessor ones using the defined keys
        model_params_only = {k: v for k, v in model_specific_best_params.items() if k not in preprocessor_param_keys}

        print(f"  - {model_type.upper()}")
        # Optional: print model params here if desired
        # print(f"    Model Params: {model_params_only}")

        # Reconstruct the model instance using the model-specific parameters
        model_instance = reconstruct_model(model_type, model_params_only)
        estimators.append((model_type, model_instance))

    if not estimators:
        print("Error: No base estimators could be reconstructed for the ensemble. Exiting.")
        sys.exit(1)

    print("\n--- Evaluating Voting Classifiers on Validation Set ---")

    # Hard Voting
    # Ensure base estimators are compatible with hard voting (return class labels) - all sklearn classifiers do
    voting_clf_hard = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    pipeline_hard = Pipeline([('preprocessor', final_preprocessor), ('classifier', voting_clf_hard)])

    print("Fitting Hard Voting pipeline...")
    pipeline_hard.fit(X_train, y_train)
    y_pred_val_hard = pipeline_hard.predict(X_val)
    f1_hard = f1_score(y_val, y_pred_val_hard, average='macro')
    print(f"Hard Voting F1 Macro: {f1_hard:.4f}")

    # Soft Voting (only if all base estimators support predict_proba)
    supports_predict_proba = all(hasattr(model, 'predict_proba') for _, model in estimators)
    best_pipeline = pipeline_hard
    best_voting_type = 'hard'
    best_voting_f1 = f1_hard

    if supports_predict_proba:
        print("\nFitting Soft Voting pipeline...")
        voting_clf_soft = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        pipeline_soft = Pipeline([('preprocessor', final_preprocessor), ('classifier', voting_clf_soft)])
        pipeline_soft.fit(X_train, y_train)
        y_pred_val_soft = pipeline_soft.predict(X_val)
        f1_soft = f1_score(y_val, y_pred_val_soft, average='macro')
        print(f"Soft Voting F1 Macro: {f1_soft:.4f}")

        if f1_soft > f1_hard:
            best_pipeline = pipeline_soft
            best_voting_type = 'soft'
            best_voting_f1 = f1_soft
            print("Soft voting performed better on validation set.")
        else:
             print("Hard voting performed better or equally well on validation set.")
    else:
        print("\nSoft voting is not supported as not all selected base estimators have 'predict_proba'.")


    print(f"\n--- Finalizing Best Voting Model ({best_voting_type.upper()}) ---")
    # The best_pipeline is already fitted on X_train, y_train with the chosen voting type

    y_pred_final_val = best_pipeline.predict(X_val)
    final_accuracy = accuracy_score(y_val, y_pred_final_val)
    print(f"Final Validation Accuracy ({best_voting_type.upper()} Voting): {final_accuracy:.4f}")
    print(f"Final Validation F1 Macro ({best_voting_type.upper()} Voting): {best_voting_f1:.4f}")

    print(f"\n--- Final Model Configuration ({best_voting_type.upper()} Voting) ---")

    print("\nPreprocessor Parameters (from overall best trial):")
    for key, value in best_preprocessor_params.items():
        print(f"  {key}: {value}")

    print("\nBase Estimators (Top {} by score) and their Tuned Parameters:".format(top_n))
    # top_models_info contains (model_type, results_dict) for the top N models
    for model_type, results in top_models_info:
        print(f"  - {model_type.upper()} (Best Val Score: {results['best_score']:.4f})")
        # Extract only the model-specific parameters
        model_params = {k: v for k, v in results['best_params'].items() if k not in preprocessor_param_keys}
        for param_name, param_value in model_params.items():
            print(f"    {param_name}: {param_value}")


    # Save components
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    # Save the best voting model type in the filename for clarity
    model_path = os.path.join(model_dir, f'trained_{best_voting_type}_voting_model.joblib')

    print(f"\nSaving fitted preprocessor to: {preprocessor_path}")
    # We save the *fitted* preprocessor
    dump(best_pipeline.named_steps['preprocessor'], preprocessor_path)


    print(f"Saving trained {best_voting_type.upper()} Voting Classifier model to: {model_path}")
    # We save the *fitted* voting classifier model
    dump(best_pipeline.named_steps['classifier'], model_path)

    print("\n--- Training Finished ---")


def run_predict(model_dir, test_input_dir, output_path):
    print("--- Starting Prediction ---")

    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')

    # Find the voting model file
    model_files = [f for f in os.listdir(model_dir) if f.startswith('trained_') and f.endswith('_voting_model.joblib')]
    if not model_files:
        print(f"Error: No trained voting model file found in {model_dir}. Expected file starting with 'trained_' and ending with '_voting_model.joblib'. Exiting.")
        sys.exit(1)
    if len(model_files) > 1:
         print(f"Warning: Found multiple voting model files in {model_dir}. Using the first one: {model_files[0]}")
    model_path = os.path.join(model_dir, model_files[0])


    if not os.path.exists(preprocessor_path):
        print(f"Error: Preprocessor file not found at {preprocessor_path}. Exiting.")
        sys.exit(1)
    # Custom Winsorizer class needs to be available in the environment where load is called
    try:
        loaded_preprocessor = load(preprocessor_path)
        print(f"Loaded preprocessor from: {preprocessor_path}")
    except Exception as e:
        print(f"Error loading preprocessor from {preprocessor_path}: {e}")
        print("Please ensure the Winsorizer class definition is available.")
        sys.exit(1)


    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Exiting.")
        sys.exit(1)
    try:
        loaded_model = load(model_path)
        print(f"Loaded model from: {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure necessary classifier classes (e.g., XGBoost, LightGBM) are installed and available.")
        sys.exit(1)


    test_path = os.path.join(test_input_dir, 'test.json')

    if not os.path.exists(test_path):
        print(f"Error: Test data file not found at {test_path}. Exiting.")
        sys.exit(1)
    df_test = pd.read_json(test_path, lines=True)

    # Drop non-predictive columns in test data as well
    cols_to_drop = [
        'id', 'name', 'first', 'last', 'compas_screening_date', 'dob',
        'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_offense_date',
        'c_arrest_date', 'screening_date', 'violent_recid', 'is_violent_recid',
        'c_case_number', 'decile_score'
    ]
    cols_to_drop_existing = [col for col in cols_to_drop if col in df_test.columns]
    if cols_to_drop_existing:
        df_test = df_test.drop(columns=cols_to_drop_existing)
        print(f"Dropped non-predictive columns in test data: {cols_to_drop_existing}")


    df_test = engineer_features(df_test)

    print(f"Loaded {len(df_test)} test cases.")
    if df_test.empty:
       print("Warning: Test data is empty.")
       # Create an empty output file with the correct format
       pd.DataFrame({'two_year_recid': []}, dtype=int).to_json(output_path, orient='records', lines=True)
       print("Empty prediction file created.")
       print("--- Prediction Finished ---")
       sys.exit(0)

    # Ensure column order consistency might be needed if ColumnTransformer had issues,
    # but typically ColumnTransformer handles column subsets correctly.
    # However, missing columns in test that were in train (other than dropped ones) could be an issue.
    # ColumnTransformer will ignore columns in test not seen in train (if not in transformers/remainder).
    # It will raise an error if a column specified in a transformer is missing.
    # We assume features are consistent between train and test after initial dropping/engineering.


    # Apply the loaded preprocessor and model
    # The preprocessor expects the same columns (after engineering) as it saw during fit.
    # The engineer_features function should be applied to test data too.

    try:
        # Preprocessing test data
        X_test_processed = loaded_preprocessor.transform(df_test)

        # Making predictions
        predictions = loaded_model.predict(X_test_processed)
        print(f"Generated {len(predictions)} predictions.")

    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Please ensure test data format and columns are compatible with the loaded preprocessor.")
        sys.exit(1)


    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save predictions in the required format (JSON lines)
    pd.DataFrame({'two_year_recid': predictions}).to_json(output_path, orient='records', lines=True)
    print(f"Predictions saved to: {output_path}")

    print("--- Prediction Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Train/Predict Recidivism Voting Ensemble Model")
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    parser_train = subparsers.add_parser('train', help='Tune models, create best voting ensemble, save preprocessor and model')
    parser_train.add_argument('--public_dir', type=str, default='./', help='Directory containing train_data/train.json')
    parser_train.add_argument('--model_dir', type=str, default ='./model_output', help='Directory to save preprocessor.joblib and trained_voting_model.joblib')
    parser_train.add_argument('--n_trials_per_model', type=int, default=15, help='Number of Optuna trials FOR EACH base model type (default: 15)')

    parser_predict = subparsers.add_parser('predict', help='Make predictions using saved preprocessor and voting model')
    parser_predict.add_argument('--model_dir', type=str, default='./model_output', help='Directory containing preprocessor.joblib and trained_..._voting_model.joblib')
    parser_predict.add_argument('--test_input_dir', type=str, default='./test_data', help='Directory containing test.json')
    parser_predict.add_argument('--output_path', type=str, default='./predict.json', help='File path for saving predictions')

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args.public_dir, args.model_dir, args.n_trials_per_model)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.test_input_dir, args.output_path)
    else:
         parser.print_help()


if __name__ == "__main__":
    # Ensure the Winsorizer class is available when loading models
    # In a single script, this is usually handled by just having the class definition present.
    # If splitting into modules, ensure Winsorizer is imported where needed.
    main()