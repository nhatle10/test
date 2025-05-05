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
from sklearn.model_selection import train_test_split, StratifiedKFold # Import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from joblib import dump, load
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        if not 0 <= lower_quantile <= 1:
            # Adjust bounds if suggested values are out of range
            if lower_quantile < 0: lower_quantile = 0
            if lower_quantile > 1: lower_quantile = 1
            # raise ValueError("lower_quantile must be between 0 and 1") # Optuna suggestions might slightly exceed boundaries
        if not 0 <= upper_quantile <= 1:
            if upper_quantile < 0: upper_quantile = 0
            if upper_quantile > 1: upper_quantile = 1
            # raise ValueError("upper_quantile must be between 0 and 1") # Optuna suggestions might slightly exceed boundaries

        # Handle potential issue where suggestion might make lower > upper
        if lower_quantile > upper_quantile:
            warnings.warn(f"lower_quantile ({lower_quantile:.4f}) > upper_quantile ({upper_quantile:.4f}). Swapping them.", UserWarning)
            self.lower_quantile = upper_quantile
            self.upper_quantile = lower_quantile
        else:
            self.lower_quantile = lower_quantile
            self.upper_quantile = upper_quantile


    def fit(self, X, y=None):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
             raise TypeError("Input X must be pandas DataFrame or numpy array")

        X_np = X if isinstance(X, np.ndarray) else X.values

        # Use nanquantile just in case, though imputation should handle NaNs before this
        self.lower_bounds_ = np.nanquantile(X_np, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X_np, self.upper_quantile, axis=0)

        return self

    def transform(self, X):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
             raise TypeError("Input X must be pandas DataFrame or numpy array")

        X_np = X if isinstance(X, np.ndarray) else X.values

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

# def engineer_features(df):
#     # Ensure columns exist before checking/dropping/engineering
#     df_eng = df.copy() # Work on a copy to avoid modifying original df unexpectedly

#     if 'age' in df_eng.columns:
#         df_eng['age_cat'] = df_eng['age'].apply(categorize_age)
#         df_eng.drop(columns=['age'], inplace=True)
#     else:
#         # Handle cases where 'age' might be missing entirely (e.g., a peculiar test set)
#         df_eng['age_cat'] = np.nan # Add the engineered column with NaNs
#         print("Warning: 'age' column not found for feature engineering, 'age_cat' added with NaNs.")


#     # Ensure columns exist before accessing and performing calculations
#     juv_fel_count = df_eng.get('juv_fel_count', pd.Series(0, index=df_eng.index))
#     juv_misd_count = df_eng.get('juv_misd_count', pd.Series(0, index=df_eng.index))
#     juv_other_count = df_eng.get('juv_other_count', pd.Series(0, index=df_eng.index))
#     priors_count = df_eng.get('priors_count', pd.Series(0, index=df_eng.index))
#     c_charge_degree = df_eng.get('c_charge_degree', pd.Series('', index=df_eng.index)) # Default to empty string or mode if preferred

#     # Fillna(0) before summing counts to ensure numeric operations
#     df_eng['juv_total_count'] = juv_fel_count.fillna(0) + juv_misd_count.fillna(0) + juv_other_count.fillna(0)

#     # Ensure sum is treated as numeric, fillna(0) if necessary before > 0 comparison
#     df_eng['is_juvenile_offender'] = (df_eng['juv_total_count'].fillna(0) > 0).astype(int)

#     # Đã từng phạm pháp trước đó?
#     df_eng['is_repeat_offender'] = (priors_count.fillna(0) > 0).astype(int)

#     # Có phạm tội nghiêm trọng không?
#     df_eng['is_felony'] = (c_charge_degree == 'F').astype(int)

#     # Xóa các cột không còn cần thiết
#     df.drop(columns=['juv_fel_count', 'juv_misd_count', 'juv_other_count'], inplace=True)


#     return df_eng

def engineer_features(df):
    # Age category
    df['age_cat'] = df['age'].apply(categorize_age)
    df.drop(columns=['age'], inplace=True)

    # Tổng số vi phạm vị thành niên
    df['juv_total_count'] = df['juv_fel_count'] + df['juv_misd_count'] + df['juv_other_count']

    # Có từng phạm pháp khi vị thành niên không?
    df['is_juvenile_offender'] = (df['juv_total_count'] > 0).astype(int)

    # Đã từng phạm pháp trước đó?
    df['is_repeat_offender'] = (df['priors_count'] > 0).astype(int)

    # Có phạm tội nghiêm trọng không?
    df['is_felony'] = (df['c_charge_degree'] == 'F').astype(int)

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
        solver = 'liblinear'
        params['logreg_class_weight'] = trial.suggest_categorical('logreg_class_weight', [None, 'balanced']) # Add class_weight tuning
        model = LogisticRegression(C=params['logreg_c'], penalty=params['logreg_penalty'], solver=solver, max_iter=2000, random_state=42, n_jobs=-1,
                                   class_weight=params['logreg_class_weight']) # Use tuned class_weight

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
        params['svc_class_weight'] = trial.suggest_categorical('svc_class_weight', [None, 'balanced']) # Add class_weight tuning

        model = SVC(C=params['svc_c'], kernel=params['svc_kernel'], gamma=gamma, degree=degree, probability=True, random_state=42,
                    class_weight=params['svc_class_weight']) # Use tuned class_weight

    elif model_type == 'decision_tree':
        params['dt_criterion'] = trial.suggest_categorical("dt_criterion", ["gini", "entropy"])
        params['dt_max_depth'] = trial.suggest_int("dt_max_depth", 5, 25)
        params['dt_min_samples_split'] = trial.suggest_int("dt_min_samples_split", 2, 10)
        params['dt_min_samples_leaf'] = trial.suggest_int("dt_min_samples_leaf", 1, 10)
        params['dt_class_weight'] = trial.suggest_categorical('dt_class_weight', [None, 'balanced']) # Add class_weight tuning

        model = DecisionTreeClassifier(
            criterion=params['dt_criterion'],
            max_depth=params['dt_max_depth'],
            min_samples_split=params['dt_min_samples_split'],
            min_samples_leaf=params['dt_min_samples_leaf'],
            random_state=42,
            class_weight=params['dt_class_weight'] # Use tuned class_weight
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
        params['rf_class_weight'] = trial.suggest_categorical('rf_class_weight', [None, 'balanced']) # Add class_weight tuning


        model = RandomForestClassifier(
            n_estimators=params['rf_n_estimators'],
            max_depth=max_depth,
            min_samples_split=params['rf_min_samples_split'],
            min_samples_leaf=params['rf_min_samples_leaf'],
            max_features=max_features,
            criterion=params['rf_criterion'],
            random_state=42,
            n_jobs=-1,
            class_weight=params['rf_class_weight'] # Use tuned class_weight
        )

    elif model_type == 'gradient_boosting':
        # Gradient Boosting does not have a class_weight parameter directly.
        # Sample_weight can be used in fit(), but tuning that via Optuna is complex.
        # Often rely on boosting nature and potential resampling in pipeline (like SMOTE)
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
        # GaussianNB does not have class_weight parameter.
        # Resampling techniques are the standard way to handle imbalance for NB.
        params['gnb_var_smoothing'] = trial.suggest_float("gnb_var_smoothing", 1e-9, 1e-5, log=True)
        model = GaussianNB(var_smoothing=params['gnb_var_smoothing'])

    elif model_type == 'mlp':
        # MLPClassifier does not have class_weight parameter directly.
        num_hidden_layers = trial.suggest_int("mlp_num_hidden_layers", 1, 2)
        layer_sizes = []
        for i in range(num_hidden_layers):
            layer_size = trial.suggest_int(f"mlp_n_neurons_layer_{i}", 50, 150)
            layer_sizes.append(layer_size)
        params['mlp_hidden_layer_sizes'] = tuple(layer_sizes)

        params['mlp_activation'] = trial.suggest_categorical("mlp_activation", ["relu", "tanh"])
        params['mlp_alpha'] = trial.suggest_float("mlp_alpha", 1e-5, 1e-3, log=True)
        params['mlp_learning_rate_init'] = trial.suggest_float("mlp_learning_rate_init", 1e-3, 1e-2, log=True)
        model = MLPClassifier(
            hidden_layer_sizes=params['mlp_hidden_layer_sizes'],
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
        # Add scale_pos_weight tuning for handling imbalance
        # Need to calculate this based on y_train_full before tuning starts or pass it
        # For tuning, we can suggest a range or use a fixed value based on the overall training data ratio
        # Let's assume we calculate scale_pos_weight_value outside and pass it
        # Or, more simply for tuning, suggest a factor that multiplies the imbalance ratio
        # A simple approach for tuning is to just enable/disable it or suggest a multiplier
        # Let's assume scale_pos_weight_value is passed or calculated based on global training data
        # params['xgb_scale_pos_weight_factor'] = trial.suggest_float('xgb_scale_pos_weight_factor', 0.5, 5.0) # Example tuning factor
        # scale_pos_weight_value = calculated_imbalance_ratio * params['xgb_scale_pos_pos_weight_factor'] # Needs imbalance ratio
        # A simpler tuning approach: suggest scale_pos_weight directly
        params['xgb_scale_pos_weight'] = trial.suggest_float('xgb_scale_pos_weight', 0.1, 10.0, log=True)


        model = xgb.XGBClassifier(
            n_estimators=params['xgb_n_estimators'],
            learning_rate=params['xgb_learning_rate'],
            max_depth=params['xgb_max_depth'],
            subsample=params['xgb_subsample'],
            colsample_bytree=params['xgb_colsample_bytree'],
            gamma=params['xgb_gamma'],
            reg_alpha=params['xgb_reg_alpha'],
            reg_lambda=params['xgb_reg_lambda'],
            scale_pos_weight=params['xgb_scale_pos_weight'], # Use tuned scale_pos_weight
            use_label_encoder=False,
            eval_metric='logloss', # Or 'auc', 'f1' (if supported directly)
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
         # Add scale_pos_weight or is_unbalance tuning for handling imbalance
        params['lgbm_scale_pos_weight'] = trial.suggest_float('lgbm_scale_pos_weight', 0.1, 10.0, log=True)


        model = lgb.LGBMClassifier(
            n_estimators=params['lgbm_n_estimators'],
            learning_rate=params['lgbm_learning_rate'],
            num_leaves=params['lgbm_num_leaves'],
            max_depth=params['lgbm_max_depth'],
            subsample=params['lgbm_subsample'],
            colsample_bytree=params['lgbm_colsample_bytree'],
            reg_alpha=params['lgbm_reg_alpha'],
            reg_lambda=params['lgbm_reg_lambda'],
            scale_pos_weight=params['lgbm_scale_pos_weight'], # Use tuned scale_pos_weight
            random_state=42,
            n_jobs=-1,
            verbose = -1,
            objective='binary', # Specify objective for binary classification
            metric='binary_logloss' # Specify metric
        )

    elif model_type == 'catboost':
        params['cb_iterations'] = trial.suggest_int("cb_iterations", 100, 300)
        params['cb_learning_rate'] = trial.suggest_float("cb_learning_rate", 1e-2, 0.2, log=True)
        params['cb_depth'] = trial.suggest_int("cb_depth", 3, 7)
        params['cb_l2_leaf_reg'] = trial.suggest_float("cb_l2_leaf_reg", 1, 10, log=True)
        params['cb_border_count'] = trial.suggest_int("cb_border_count", 32, 128)
        params['cb_random_strength'] = trial.suggest_float('cb_random_strength', 0.1, 2.0, log=True)
        # CatBoost has auto-handling for imbalance or class_weights/scale_pos_weight
        # params['cb_scale_pos_weight'] = trial.suggest_float('cb_scale_pos_weight', 0.1, 10.0, log=True)


        model = CatBoostClassifier(
            iterations=params['cb_iterations'],
            learning_rate=params['cb_learning_rate'],
            depth=params['cb_depth'],
            l2_leaf_reg=params['cb_l2_leaf_reg'],
            border_count=params['cb_border_count'],
            random_strength=params['cb_random_strength'],
            # scale_pos_weight=params.get('cb_scale_pos_weight', 1.0), # Use tuned or default
            random_state=42,
            verbose=0,
            thread_count=-1,
            eval_metric='F1', # Use F1 as evaluation metric during tuning
            objective='Binary:logistic' # Specify objective
        )


    elif model_type == 'adaboost':
        params['ada_base_max_depth'] = trial.suggest_int("ada_base_max_depth", 1, 4)
        base_estimator = DecisionTreeClassifier(max_depth=params['ada_base_max_depth'], random_state=42) # Class weight can be set here too, but usually handled by AdaBoost's re-weighting

        params['ada_n_estimators'] = trial.suggest_int("ada_n_estimators", 50, 200)
        params['ada_learning_rate'] = trial.suggest_float("ada_learning_rate", 0.01, 1.0, log=True)

        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=params['ada_n_estimators'],
            learning_rate=params['ada_learning_rate'],
            random_state=42
        )

    return model, params


# Update the objective_function signature to accept full training data
def objective_function(trial, X_train_full, y_train_full, num_features, cat_features, model_type):
    """Optuna objective function to train and evaluate a pipeline using K-Fold CV."""

    # Define all potential preprocessor parameters
    # Ensure all parameters suggested in this function are listed here
    preprocessor_param_keys = [
        'num_imputer_strategy', 'knn_imputer_n_neighbors', 'apply_log_transform',
        'winsorizer_lower_quantile', 'winsorizer_upper_quantile',
        'scaler_type', 'cat_imputer_strategy', 'ohe_handle_unknown', 'ohe_min_frequency'
    ]

    # --- Preprocessing Hyperparameters (Suggested Once Per Trial) ---
    num_imputer_strategy = trial.suggest_categorical('num_imputer_strategy', ['mean', 'median', 'knn'])
    knn_imputer_n_neighbors = trial.suggest_int('knn_imputer_n_neighbors', 5, 20) if num_imputer_strategy == 'knn' else None

    cat_imputer_strategy = trial.suggest_categorical('cat_imputer_strategy', ['most_frequent', 'constant'])
    cat_imputer_fill_value = 'missing' if cat_imputer_strategy == 'constant' else None

    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'robust'])
    scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()

    apply_log_transform = trial.suggest_categorical('apply_log_transform', [True, False])

    # Winsorizer quantiles - TUNABLE
    winsorizer_lower_quantile = trial.suggest_float('winsorizer_lower_quantile', 0.001, 0.1)
    winsorizer_upper_quantile = trial.suggest_float('winsorizer_upper_quantile', 0.9, 0.999)

    # One-Hot Encoding parameters
    ohe_handle_unknown = trial.suggest_categorical('ohe_handle_unknown', ['ignore', 'infrequent_if_exist'])
    ohe_min_frequency = trial.suggest_float('ohe_min_frequency', 0.005, 0.05) if ohe_handle_unknown == 'infrequent_if_exist' else None

    # --- Model and its Hyperparameters (Suggested Once Per Trial) ---
    model, model_params = get_model_and_params(trial, model_type) # Get model instance and its specific params. Params dict includes model-specific suggested values.

    # --- K-Fold Cross-Validation Setup ---
    n_splits = 5 # Define the number of folds for CV
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # Use StratifiedKFold for classification
    fold_scores = []

    # --- Perform CV for the current trial's parameters ---
    # Wrap the loop in a try/except block to handle errors during CV folds
    try:
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
            # Extract data for the current fold
            X_train_fold = X_train_full.iloc[train_idx].copy() # Use .copy() to be safe
            y_train_fold = y_train_full.iloc[train_idx].copy()
            X_val_fold = X_train_full.iloc[val_idx].copy()
            y_val_fold = y_train_full.iloc[val_idx].copy()

            # --- Build Pipeline for the current fold using the trial's parameters ---
            # Rebuild the preprocessor pipeline steps for this fold's data
            numeric_steps = []
            if num_imputer_strategy == 'knn':
                numeric_steps.append(('imputer', KNNImputer(n_neighbors=knn_imputer_n_neighbors)))
            else:
                numeric_steps.append(('imputer', SimpleImputer(strategy=num_imputer_strategy)))

            numeric_steps.append(('winsorizer', Winsorizer(lower_quantile=winsorizer_lower_quantile, upper_quantile=winsorizer_upper_quantile)))

            if apply_log_transform:
                numeric_steps.append(('log_transform', FunctionTransformer(func=np.log1p, validate=False)))

            numeric_steps.append(('scaler', scaler)) # Use the created scaler instance

            numeric_transformer = Pipeline(steps=numeric_steps)

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=cat_imputer_strategy, fill_value=cat_imputer_fill_value)),
                ('onehot', OneHotEncoder(handle_unknown=ohe_handle_unknown, min_frequency=ohe_min_frequency, sparse_output=False))
            ])

            fold_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, num_features),
                    ('cat', categorical_transformer, cat_features)
                ],
                remainder='passthrough',
                n_jobs=-1
            )

            # Reconstruct the model instance for this fold using the model-specific tuned parameters
            # We need the full trial.params here to reconstruct complex models like MLP or AdaBoost base
            fold_model = reconstruct_model(model_type, trial.params)


            # Create the full pipeline for this fold
            fold_pipeline = Pipeline(steps=[('preprocessor', fold_preprocessor), ('classifier', fold_model)])

            # --- Fit and Evaluate on the current fold ---
            fold_pipeline.fit(X_train_fold, y_train_fold)
            y_pred_val_fold = fold_pipeline.predict(X_val_fold)
            fold_score = f1_score(y_val_fold, y_pred_val_fold, average='macro')
            fold_scores.append(fold_score)

            # Report intermediate score after each fold for pruning
            # Reporting the average up to the current fold can help the pruner make better decisions
            trial.report(np.mean(fold_scores), step=fold)

            # Prune trial if needed based on intermediate performance
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Calculate the average score across all folds after the loop completes
        avg_score = np.mean(fold_scores)

    except Exception as e:
        # If an error occurs in any fold, prune the trial
        print(f"Trial {trial.number} ({model_type}) failed during CV (Fold {fold+1} error: {e}). Pruning trial.")
        # It's important to prune or return a very low score (-inf) if an error happens,
        # otherwise Optuna might think a partially completed trial is good.
        # Raising TrialPruned is the standard way to handle errors that invalidate a trial.
        raise optuna.TrialPruned()


    # Return the average score across all folds
    return avg_score


def reconstruct_model(model_type, trial_params): # Accept trial_params now to get both model and some preproc info if needed (e.g., AdaBoost base depth)
    """Reconstruct a model instance from its type and the full trial parameters dictionary."""
    model = None
    # Define preprocessor keys here too, to exclude them when getting model-specific params
    preprocessor_param_keys = [
        'num_imputer_strategy', 'knn_imputer_n_neighbors', 'apply_log_transform',
        'winsorizer_lower_quantile', 'winsorizer_upper_quantile',
        'scaler_type', 'cat_imputer_strategy', 'ohe_handle_unknown', 'ohe_min_frequency',
        # Also exclude calculated/internal keys
         '_svc_calculated_gamma', '_svc_calculated_degree',
         '_rf_calculated_max_depth', '_rf_calculated_max_features',
         'mlp_num_hidden_layers', # Exclude the layer count itself
    ]
    # Extract model-specific params from the full trial_params
    model_params = {k: v for k, v in trial_params.items() if k not in preprocessor_param_keys}

    if model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=model_params.get('knn_n_neighbors', 5),
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
            n_jobs=-1,
            class_weight=model_params.get('logreg_class_weight', None) # Use tuned class_weight
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
            random_state=42,
            class_weight=model_params.get('svc_class_weight', None) # Use tuned class_weight
        )
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            criterion=model_params.get('dt_criterion', 'gini'),
            max_depth=model_params.get('dt_max_depth', 10),
            min_samples_split=model_params.get('dt_min_samples_split', 2),
            min_samples_leaf=model_params.get('dt_min_samples_leaf', 1),
            random_state=42,
            class_weight=model_params.get('dt_class_weight', None) # Use tuned class_weight
        )
    elif model_type == 'random_forest':
         max_depth_val = model_params.get('_rf_calculated_max_depth')
         max_features_val = model_params.get('_rf_calculated_max_features')
         model = RandomForestClassifier(
            n_estimators=model_params.get('rf_n_estimators', 100),
            max_depth=max_depth_val,
            min_samples_split=model_params.get('rf_min_samples_split', 2),
            min_samples_leaf=model_params.get('rf_min_samples_leaf', 1),
            max_features=max_features_val,
            criterion=model_params.get('rf_criterion', 'gini'),
            random_state=49,
            n_jobs=-1,
            class_weight=model_params.get('rf_class_weight', None) # Use tuned class_weight
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
        # Retrieve hidden layer sizes based on how they were suggested
        num_hidden_layers = trial_params.get("mlp_num_hidden_layers", 1) # Need this from full params
        layer_sizes = []
        for i in range(num_hidden_layers):
             layer_sizes.append(trial_params.get(f"mlp_n_neurons_layer_{i}", 100))
        hidden_layer_sizes_tuple = tuple(layer_sizes)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes_tuple,
            activation=model_params.get('mlp_activation', 'relu'),
            solver='adam',
            alpha=model_params.get('mlp_alpha', 0.0001),
            learning_rate_init=model_params.get('mlp_learning_rate_init', 0.001),
            max_iter=1500,
            early_stopping=False, # Turn off early stopping for final training (used in objective)
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
            scale_pos_weight=model_params.get('xgb_scale_pos_weight', 1.0), # Use tuned or default
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
            max_depth=model_params.get('lgbm_max_depth', -1),
            subsample=model_params.get('lgbm_subsample', 1.0),
            colsample_bytree=model_params.get('lgbm_colsample_bytree', 1.0),
            reg_alpha=model_params.get('lgbm_reg_alpha', 0),
            reg_lambda=model_params.get('lgbm_reg_lambda', 0),
            scale_pos_weight=model_params.get('lgbm_scale_pos_weight', 1.0), # Use tuned or default
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            objective='binary',
            metric='binary_logloss'
        )
    elif model_type == 'catboost':
        model = CatBoostClassifier(
            iterations=model_params.get('cb_iterations', 100),
            learning_rate=model_params.get('cb_learning_rate', 0.1),
            depth=model_params.get('cb_depth', 6),
            l2_leaf_reg=model_params.get('cb_l2_leaf_reg', 3),
            border_count=model_params.get('cb_border_count', 254),
            random_strength=model_params.get('cb_random_strength', 1.0),
            # scale_pos_weight=model_params.get('cb_scale_pos_weight', 1.0), # Use tuned or default
            random_state=42,
            verbose=0,
            thread_count=-1,
            eval_metric='F1',
            objective='Binary:logistic'
        )
    elif model_type == 'adaboost':
        base_estimator_max_depth = model_params.get('ada_base_max_depth', 1)
        base_estimator = DecisionTreeClassifier(max_depth=base_estimator_max_depth, random_state=42)

        model = AdaBoostClassifier(
            estimator=base_estimator,
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
    # Use .get with default values in case a parameter key is missing (e.g., in older trial params)
    num_imputer_strategy = preprocessor_params.get('num_imputer_strategy', 'mean')
    knn_imputer_n_neighbors = preprocessor_params.get('knn_imputer_n_neighbors', 5)

    cat_imputer_strategy = preprocessor_params.get('cat_imputer_strategy', 'most_frequent')
    cat_imputer_fill_value = 'missing' if cat_imputer_strategy == 'constant' else None

    scaler_type = preprocessor_params.get('scaler_type', 'standard')
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else: # robust
        scaler = RobustScaler()

    apply_log_transform = preprocessor_params.get('apply_log_transform', False)

    # Get TUNED Winsorizer parameters, default to 0.01, 0.99 if not found
    winsorizer_lower_quantile = preprocessor_params.get('winsorizer_lower_quantile', 0.01)
    winsorizer_upper_quantile = preprocessor_params.get('winsorizer_upper_quantile', 0.99)


    ohe_handle_unknown = preprocessor_params.get('ohe_handle_unknown', 'ignore')
    ohe_min_frequency = preprocessor_params.get('ohe_min_frequency', None)


    # Build numeric pipeline steps using the best parameters
    numeric_steps = []
    # 1. Imputation
    if num_imputer_strategy == 'knn':
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


def run_train(public_dir, model_dir, n_trials_per_model=15):
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
    cols_to_drop = [
        'id', 'name', 'first', 'last', 'compas_screening_date', 'dob',
        'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_offense_date',
        'c_arrest_date', 'screening_date', 'violent_recid', 'is_violent_recid',
        'c_case_number', 'decile_score'
    ]
    cols_to_drop_existing = [col for col in cols_to_drop if col in X.columns]
    if cols_to_drop_existing:
        X = X.drop(columns=cols_to_drop_existing)
        print(f"Dropped non-predictive columns: {cols_to_drop_existing}")


    X = engineer_features(X) # Engineer features *before* splitting

    # Determine features after engineering and dropping columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # --- Initial Split into Training (for CV Tuning) and Final Validation Sets ---
    # We split the original data into a larger 'full training' set for Optuna's CV
    # and a smaller 'final validation' set that is completely held out until the end.
    if len(y.unique()) < 2:
         print("Error: Target variable does not have at least 2 classes after data loading. Cannot train. Exiting.")
         sys.exit(1)

    min_samples_per_class = y.value_counts().min()
    if min_samples_per_class < 2:
         print(f"Warning: Minimum samples per class is {min_samples_per_class}. Stratification may not be possible or reliable for the initial split.")
         # Attempt stratification anyway, as train_test_split handles cases where a class has 1 sample
         # It might put that sample in either train or test, but won't raise error unless min_samples < n_splits
         # For initial split, let's keep stratification if possible
         try:
             X_train_full, X_val_final, y_train_full, y_val_final = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
             )
             print("Initial split using stratification.")
         except ValueError as e:
             print(f"Stratified split failed ({e}). Falling back to non-stratified split.")
             X_train_full, X_val_final, y_train_full, y_val_final = train_test_split(
                X, y, test_size=0.25, random_state=42
             )
    else:
        X_train_full, X_val_final, y_train_full, y_val_final = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        print("Initial split using stratification.")


    print(f"Full Training set size (for CV tuning): {len(X_train_full)}")
    print(f"Final Held-out Validation set size: {len(X_val_final)}")


    supported_models = [
        'knn', 'logistic', 'svc', 'decision_tree', 'random_forest',
        #'gradient_boosting', 'gaussian_nb', 'mlp',
        'xgboost', 'lightgbm', 'catboost', 'adaboost'
    ]
    all_model_results = {}
    studies = {}

    print(f"\n--- Starting Optuna Hyperparameter Tuning ({n_trials_per_model} Trials Per Model using CV) ---")

    for model_type in tqdm(supported_models, desc="Model Types Tuned"):
        print(f"\nTuning {model_type.upper()}...")

        # Create a new study for each model type
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

        # Create a partial function with fixed arguments for the objective
        # Pass the full training data (X_train_full, y_train_full) to the objective
        objective_func = lambda trial: objective_function(
            trial, X_train_full, y_train_full, num_features, cat_features, model_type
        )

        try:
            study.optimize(
                objective_func,
                n_trials=n_trials_per_model,
                n_jobs=-1,
                show_progress_bar=False
            )

            # Store results for models where tuning completed successfully (best_value is not None)
            if study.best_trial and study.best_trial.value is not None:
                all_model_results[model_type] = {
                    'best_score_cv': study.best_value, # Best score is now the average CV score
                    'best_params': study.best_trial.params
                }
                studies[model_type] = study
                print(f"  Best Avg CV F1 Macro for {model_type.upper()}: {study.best_value:.4f}")
            else:
                 print(f"  Tuning failed or yielded no valid trials for {model_type.upper()}.")


        except Exception as e:
            print(f"An unhandled error occurred during tuning for {model_type.upper()}: {e}")


    print(f"\n--- Optuna Tuning Finished ---")

    # Filter out models that didn't yield valid tuning results
    # Check specifically for 'best_score_cv' key which indicates success
    valid_model_results = {k: v for k, v in all_model_results.items() if 'best_score_cv' in v}

    if not valid_model_results:
        print("Error: Optuna tuning failed or produced no valid results for all tested model types. Cannot ensemble. Exiting.")
        sys.exit(1)

    # Sort models by their best average CV score
    sorted_models = sorted(valid_model_results.items(), key=lambda item: item[1]['best_score_cv'], reverse=True)

    # Determine how many top models to ensemble
    top_n = 3
    if len(sorted_models) < top_n:
        print(f"Warning: Fewer than {top_n} models had valid tuning results ({len(sorted_models)}). Using all available valid models.")
        top_n = len(sorted_models)

    if top_n == 0:
        print("Error: No valid models available to form an ensemble after tuning. Exiting.")
        sys.exit(1)


    top_models_info = sorted_models[:top_n]

    print("\n--- Top Performing Models based on Average CV Score ---")
    for i, (model_type, results) in enumerate(top_models_info):
        print(f"{i+1}. Model: {model_type.upper()}, Best Avg CV F1 Macro: {results['best_score_cv']:.4f}")


    # Use the preprocessor parameters from the single overall best trial across ALL models
    # This trial's parameter set achieved the highest average CV score.
    overall_best_trial_across_models = None
    overall_best_score_cv = -float('inf')

    for model_type, study in studies.items():
        # Check if study exists and has a best trial with a valid score
        if study and study.best_trial and study.best_trial.value is not None:
             if study.best_trial.value > overall_best_score_cv:
                 overall_best_score_cv = study.best_trial.value
                 overall_best_trial_across_models = study.best_trial

    if overall_best_trial_across_models is None:
        # This shouldn't happen if valid_model_results is not empty, but added for safety
        print("Error: Could not identify a single best trial across all studies with a valid score. Cannot build preprocessor. Exiting.")
        sys.exit(1)


    print(f"\nUsing preprocessor settings from the overall best trial (Avg CV score: {overall_best_score_cv:.4f}).")
    # Define all preprocessor parameter keys
    preprocessor_param_keys = [
        'num_imputer_strategy', 'knn_imputer_n_neighbors', 'apply_log_transform',
        'winsorizer_lower_quantile', 'winsorizer_upper_quantile',
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
    print("\nReconstructing Top Models for Ensemble (using their best trial params):")
    for model_type, results in top_models_info:
        # Get the best trial parameters for *this specific model type*
        model_specific_best_params = results['best_params']

        # Reconstruct the model instance using the full parameter dictionary from its best trial
        # The reconstruct_model function handles extracting model-specific params internally
        print(f"  - {model_type.upper()} (Best Avg CV Score: {results['best_score_cv']:.4f})")
        model_instance = reconstruct_model(model_type, model_specific_best_params)
        estimators.append((model_type, model_instance))

    if not estimators:
        print("Error: No base estimators could be reconstructed for the ensemble. Exiting.")
        sys.exit(1)

    print("\n--- Building and Fitting Final Voting Ensemble ---")
    # Hard Voting
    voting_clf_hard = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    pipeline_hard = Pipeline([('preprocessor', final_preprocessor), ('classifier', voting_clf_hard)])

    print("Fitting Hard Voting pipeline on full training data...")
    pipeline_hard.fit(X_train_full, y_train_full) # Fit the final ensemble on the full training data


    # Soft Voting (only if all base estimators support predict_proba)
    supports_predict_proba = all(hasattr(model, 'predict_proba') for _, model in estimators)
    best_pipeline = pipeline_hard
    best_voting_type = 'hard'


    if supports_predict_proba:
        print("\nFitting Soft Voting pipeline on full training data...")
        voting_clf_soft = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        pipeline_soft = Pipeline([('preprocessor', final_preprocessor), ('classifier', voting_clf_soft)])
        pipeline_soft.fit(X_train_full, y_train_full) # Fit on the full training data

        # --- Evaluate Hard vs Soft Voting on the FINAL held-out validation set ---
        print("\n--- Evaluating Hard vs Soft Voting on the FINAL held-out Validation Set ---")
        y_pred_val_hard = pipeline_hard.predict(X_val_final)
        f1_hard = f1_score(y_val_final, y_pred_val_hard, average='macro')
        print(f"Hard Voting F1 Macro on final validation set: {f1_hard:.4f}")

        y_pred_val_soft = pipeline_soft.predict(X_val_final)
        f1_soft = f1_score(y_val_final, y_pred_val_soft, average='macro')
        print(f"Soft Voting F1 Macro on final validation set: {f1_soft:.4f}")


        if f1_soft > f1_hard:
            best_pipeline = pipeline_soft
            best_voting_type = 'soft'
            print("Soft voting performed better on final validation set.")
        else:
             print("Hard voting performed better or equally well on final validation set.")
    else:
        print("\nSoft voting is not supported as not all selected base estimators have 'predict_proba'.")
        # Evaluate the chosen Hard voting model on the final validation set
        print("\n--- Evaluating Hard Voting on the FINAL held-out Validation Set ---")
        y_pred_val_hard = pipeline_hard.predict(X_val_final)
        f1_hard = f1_score(y_val_final, y_pred_val_hard, average='macro')
        print(f"Hard Voting F1 Macro on final validation set: {f1_hard:.4f}")


    # --- Final Evaluation on the Held-out Validation Set ---
    print(f"\n--- Final Evaluation on Held-out Validation Set ({best_voting_type.upper()}) ---")
    # The best_pipeline is already fitted on X_train_full, y_train_full
    y_pred_final_val = best_pipeline.predict(X_val_final) # Use the final held-out set
    final_accuracy = accuracy_score(y_val_final, y_pred_final_val)
    final_f1_macro = f1_score(y_val_final, y_pred_final_val, average='macro') # This is the score used to pick hard vs soft
    print(f"Final Held-out Validation Accuracy ({best_voting_type.upper()} Voting): {final_accuracy:.4f}")
    print(f"Final Held-out Validation F1 Macro ({best_voting_type.upper()} Voting): {final_f1_macro:.4f}")


    print(f"\n--- Final Model Configuration ({best_voting_type.upper()} Voting) ---")

    print("\nPreprocessor Parameters (from overall best trial):")
    for key, value in best_preprocessor_params.items():
        print(f"  {key}: {value}")

    print("\nBase Estimators (Top {} by Avg CV Score) and their Tuned Parameters:".format(top_n))
    for model_type, results in top_models_info:
        print(f"  - {model_type.upper()} (Best Avg CV Score: {results['best_score_cv']:.4f})")
        # Print only the model-specific parameters from this model's best trial
        model_params_only = {k: v for k, v in results['best_params'].items() if k not in preprocessor_param_keys}
        for param_name, param_value in model_params_only.items():
            print(f"    {param_name}: {param_value}")


    # Save components
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    model_path = os.path.join(model_dir, f'trained_{best_voting_type}_voting_model.joblib')

    print(f"\nSaving fitted preprocessor to: {preprocessor_path}")
    # We save the *fitted* preprocessor from the best_pipeline (which was fit on X_train_full)
    dump(best_pipeline.named_steps['preprocessor'], preprocessor_path)


    print(f"Saving trained {best_voting_type.upper()} Voting Classifier model to: {model_path}")
    # We save the *fitted* voting classifier model from the best_pipeline
    dump(best_pipeline.named_steps['classifier'], model_path)

    print("\n--- Training Finished ---")


def run_predict(model_dir, test_input_dir, output_path):
    print("--- Starting Prediction ---")

    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')

    # Find the voting model file (either hard or soft)
    model_files = [f for f in os.listdir(model_dir) if f.startswith('trained_') and f.endswith('_voting_model.joblib')]
    if not model_files:
        print(f"Error: No trained voting model file found in {model_dir}. Expected file starting with 'trained_' and ending with '_voting_model.joblib'. Exiting.")
        sys.exit(1)
    # Sort to get a consistent order if multiple files exist (e.g., trained_hard... trained_soft...)
    model_files.sort()
    model_path = os.path.join(model_dir, model_files[0])
    print(f"Using model file: {model_path}")


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
        print("Please ensure necessary classifier classes (e.g., XGBoost, LightGBM, CatBoost) are installed and available.")
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
    cols_to_drop_existing = [col for col in df_test.columns if col in cols_to_drop] # Check if column exists before dropping
    if cols_to_drop_existing:
        df_test = df_test.drop(columns=cols_to_drop_existing)
        print(f"Dropped non-predictive columns in test data: {cols_to_drop_existing}")


    # Engineer features in test data - MUST be consistent with train
    df_test = engineer_features(df_test)

    print(f"Loaded {len(df_test)} test cases.")
    if df_test.empty:
       print("Warning: Test data is empty.")
       # Create an empty output file with the correct format
       pd.DataFrame({'two_year_recid': []}, dtype=int).to_json(output_path, orient='records', lines=True)
       print("Empty prediction file created.")
       print("--- Prediction Finished ---")
       sys.exit(0)

    try:
        # Preprocessing test data using the loaded preprocessor
        # The preprocessor expects the same columns (after engineering) as it saw during fit on X_train_full
        X_test_processed = loaded_preprocessor.transform(df_test)

        # Making predictions using the loaded model
        predictions = loaded_model.predict(X_test_processed)
        print(f"Generated {len(predictions)} predictions.")

    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Please ensure test data format and columns (after engineering) are compatible with the loaded preprocessor.")
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

    parser_train = subparsers.add_parser('train', help='Tune models (using CV), create best voting ensemble, save preprocessor and model')
    parser_train.add_argument('--public_dir', type=str, default='./', help='Directory containing train_data/train.json')
    parser_train.add_argument('--model_dir', type=str, default ='./model_output', help='Directory to save preprocessor.joblib and trained_voting_model.joblib')
    parser_train.add_argument('--n_trials_per_model', type=int, default=30, help='Number of Optuna trials FOR EACH base model type (default: 15)')

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
    main()