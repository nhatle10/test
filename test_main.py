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


from sklearn.impute import SimpleImputer, KNNImputer # Added KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer # Added FunctionTransformer
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
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
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
        # Use 'l2' and 'none' penalties with 'sag' or 'saga' solver for larger datasets
        # or stick to 'l1'/'l2' with 'liblinear' for potentially faster tuning on smaller datasets
        params['logreg_c'] = trial.suggest_float("logreg_c", 1e-2, 1e1, log=True)
        params['logreg_penalty'] = trial.suggest_categorical('logreg_penalty', ['l2']) # Often l2 is sufficient and more stable
        # Using 'saga' or 'liblinear' solvers. 'saga' supports L1/L2/Elastic-Net, 'liblinear' supports L1/L2 (binary only, good for small data)
        # Let's stick to 'liblinear' as it was used before and is good for binary L1/L2
        solver = 'liblinear'
        if params['logreg_penalty'] == 'l2': # liblinear doesn't support 'none'
             # For L2 with liblinear, C is effectively regularization strength inverse
             pass
        # If 'l1' was an option, liblinear supports it.

        model = LogisticRegression(C=params['logreg_c'], penalty=params['logreg_penalty'], solver=solver, max_iter=2000, random_state=42, n_jobs=-1) # Increased max_iter

    elif model_type == 'svc':
        params['svc_c'] = trial.suggest_float("svc_c", 1e-1, 1e2, log=True)
        params['svc_kernel'] = trial.suggest_categorical("svc_kernel", ["linear", "rbf"])
        gamma = "scale"
        degree = 3 # Kept for consistency, though only used by poly kernel (which is removed)

        if params['svc_kernel'] == "rbf":
            params['svc_gamma_type'] = trial.suggest_categorical("svc_gamma_type", ["scale", "float"])
            if params['svc_gamma_type'] == "float":
                params['svc_gamma_float'] = trial.suggest_float("svc_gamma_float", 1e-3, 1.0, log=True)
                gamma = params['svc_gamma_float']
            else:
                gamma = params['svc_gamma_type'] # scale

        params['_svc_calculated_gamma'] = gamma
        params['_svc_calculated_degree'] = degree # Store for info, even if not used in kernel choices

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
        params['_rf_calculated_max_depth'] = max_depth # Store calculated value
        params['_rf_calculated_max_features'] = max_features # Store calculated value


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
        # Increased the number of hidden layers option
        num_hidden_layers = trial.suggest_int("mlp_num_hidden_layers", 1, 2)
        layer_sizes = []
        for i in range(num_hidden_layers):
            layer_size = trial.suggest_int(f"mlp_n_neurons_layer_{i}", 50, 150) # Reduced range
            layer_sizes.append(layer_size)
        params['mlp_hidden_layer_sizes'] = tuple(layer_sizes)


        params['mlp_activation'] = trial.suggest_categorical("mlp_activation", ["relu", "tanh"])
        # mlp_solver = 'adam' # Sticking with 'adam' based on typical performance
        params['mlp_alpha'] = trial.suggest_float("mlp_alpha", 1e-5, 1e-3, log=True)
        params['mlp_learning_rate_init'] = trial.suggest_float("mlp_learning_rate_init", 1e-3, 1e-2, log=True)
        model = MLPClassifier(
            hidden_layer_sizes=params['mlp_hidden_layer_sizes'],
            activation=params['mlp_activation'],
            solver='adam', # Fixed solver
            alpha=params['mlp_alpha'],
            learning_rate_init=params['mlp_learning_rate_init'],
            max_iter=1500, # Increased max_iter for training
            early_stopping=True, # Use early stopping during tuning
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
        params['xgb_reg_alpha'] = trial.suggest_float('xgb_reg_alpha', 1e-2, 10.0, log=True) # Added L1 reg
        params['xgb_reg_lambda'] = trial.suggest_float('xgb_reg_lambda', 1e-2, 10.0, log=True) # Added L2 reg

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
        params['lgbm_reg_alpha'] = trial.suggest_float('lgbm_reg_alpha', 1e-2, 10.0, log=True) # Added L1 reg
        params['lgbm_reg_lambda'] = trial.suggest_float('lgbm_reg_lambda', 1e-2, 10.0, log=True) # Added L2 reg

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
        params['cb_random_strength'] = trial.suggest_float('cb_random_strength', 0.1, 2.0, log=True) # Added random strength

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
        # Tune base estimator max_depth
        params['ada_base_max_depth'] = trial.suggest_int("ada_base_max_depth", 1, 4) # Base is usually shallow tree
        base_estimator = DecisionTreeClassifier(max_depth=params['ada_base_max_depth'], random_state=42)

        params['ada_n_estimators'] = trial.suggest_int("ada_n_estimators", 50, 200)
        params['ada_learning_rate'] = trial.suggest_float("ada_learning_rate", 0.01, 1.0, log=True)

        model = AdaBoostClassifier(
            estimator=base_estimator, # Use the tuned base estimator
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
    # If KNN imputation is chosen, suggest n_neighbors
    if num_imputer_strategy == 'knn':
        knn_imputer_n_neighbors = trial.suggest_int('knn_imputer_n_neighbors', 5, 20) # Range for KNN n_neighbors
    else:
        knn_imputer_n_neighbors = None # Default if not using KNN

    # Imputation strategy for categorical features
    cat_imputer_strategy = trial.suggest_categorical('cat_imputer_strategy', ['most_frequent', 'constant'])
    cat_imputer_fill_value = 'missing' if cat_imputer_strategy == 'constant' else None

    # Scaling type
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'robust']) # Added MinMaxScaler back as an option? No, stick to standard/robust as requested.
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else: # robust
        scaler = RobustScaler()

    # Log transformation for numerical features
    apply_log_transform = trial.suggest_categorical('apply_log_transform', [True, False])

    # One-Hot Encoding parameters
    ohe_handle_unknown = trial.suggest_categorical('ohe_handle_unknown', ['ignore', 'infrequent_if_exist'])
    # Suggest min_frequency only if infrequent_if_exist is chosen
    ohe_min_frequency = trial.suggest_float('ohe_min_frequency', 0.005, 0.05) if ohe_handle_unknown == 'infrequent_if_exist' else None # Reduced range slightly, down to 0.5% frequency


    # --- Build Preprocessing Pipelines ---
    numeric_steps = []
    # 1. Imputation
    if num_imputer_strategy == 'knn':
        numeric_steps.append(('imputer', KNNImputer(n_neighbors=knn_imputer_n_neighbors)))
    else:
        numeric_steps.append(('imputer', SimpleImputer(strategy=num_imputer_strategy)))

    # 2. Log Transformation (Apply after imputation)
    if apply_log_transform:
        # Use np.log1p which computes log(1+x) to handle zeros gracefully
        numeric_steps.append(('log_transform', FunctionTransformer(func=np.log1p, validate=False))) # validate=False can prevent issues with non-finite values if they somehow occur

    # 3. Scaling (Apply after imputation and transformation)
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
    model, model_params = get_model_and_params(trial, model_type) # Get model instance and its specific params

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
        # Catch errors during fitting or prediction (e.g., due to specific hyperparameter combinations)
        # and report a low score to discourage Optuna from exploring this path.
        # print(f"Trial {trial.number} failed with error: {e}") # Optional: print error
        score = -1.0 # Report a very low score on failure
        trial.report(score, step=0) # Report failure score

    return score

def reconstruct_model(model_type, model_params):
    """Reconstruct a model instance from its type and parameters dictionary."""
    model = None
    if model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=model_params['knn_n_neighbors'],
            weights=model_params['knn_weights'],
            metric=model_params['knn_metric'],
            n_jobs=-1
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=model_params['logreg_c'],
            penalty=model_params['logreg_penalty'],
            solver='liblinear',
            max_iter=2000,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'svc':
         gamma_val = model_params.get('_svc_calculated_gamma', 'scale')
         degree_val = model_params.get('_svc_calculated_degree', 3)
         model = SVC(
            C=model_params['svc_c'],
            kernel=model_params['svc_kernel'],
            gamma=gamma_val,
            degree=degree_val,
            probability=True,
            random_state=42
        )
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            criterion=model_params['dt_criterion'],
            max_depth=model_params['dt_max_depth'],
            min_samples_split=model_params['dt_min_samples_split'],
            min_samples_leaf=model_params['dt_min_samples_leaf'],
            random_state=42
        )
    elif model_type == 'random_forest':
         max_depth_val = model_params.get('_rf_calculated_max_depth')
         max_features_val = model_params.get('_rf_calculated_max_features')
         model = RandomForestClassifier(
            n_estimators=model_params['rf_n_estimators'],
            max_depth=max_depth_val,
            min_samples_split=model_params['rf_min_samples_split'],
            min_samples_leaf=model_params['rf_min_samples_leaf'],
            max_features=max_features_val,
            criterion=model_params['rf_criterion'],
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=model_params['gb_n_estimators'],
            learning_rate=model_params['gb_learning_rate'],
            max_depth=model_params['gb_max_depth'],
            subsample=model_params['gb_subsample'],
            min_samples_split=model_params['gb_min_samples_split'],
            min_samples_leaf=model_params['gb_min_samples_leaf'],
            random_state=42
        )
    elif model_type == 'gaussian_nb':
        model = GaussianNB(
            var_smoothing=model_params['gnb_var_smoothing']
        )
    elif model_type == 'mlp':
        # --- START FIX FOR MLP HIDDEN LAYERS ---
        # Reconstruct the hidden_layer_sizes tuple from the suggested parameters
        num_hidden_layers = model_params['mlp_num_hidden_layers']
        layer_sizes = []
        for i in range(num_hidden_layers):
             # Use the correct suggested parameter key
             layer_sizes.append(model_params[f"mlp_n_neurons_layer_{i}"])
        hidden_layer_sizes_tuple = tuple(layer_sizes)
        # --- END FIX FOR MLP HIDDEN LAYERS ---

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes_tuple, # Use the reconstructed tuple
            activation=model_params['mlp_activation'],
            solver='adam',
            alpha=model_params['mlp_alpha'],
            learning_rate_init=model_params['mlp_learning_rate_init'],
            max_iter=1500,
            early_stopping=False,
            random_state=42
        )
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=model_params['xgb_n_estimators'],
            learning_rate=model_params['xgb_learning_rate'],
            max_depth=model_params['xgb_max_depth'],
            subsample=model_params['xgb_subsample'],
            colsample_bytree=model_params['xgb_colsample_bytree'],
            gamma=model_params['xgb_gamma'],
            reg_alpha=model_params['xgb_reg_alpha'],
            reg_lambda=model_params['xgb_reg_lambda'],
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators=model_params['lgbm_n_estimators'],
            learning_rate=model_params['lgbm_learning_rate'],
            num_leaves=model_params['lgbm_num_leaves'],
            max_depth=model_params['lgbm_max_depth'],
            subsample=model_params['lgbm_subsample'],
            colsample_bytree=model_params['lgbm_colsample_bytree'],
            reg_alpha=model_params['lgbm_reg_alpha'],
            reg_lambda=model_params['lgbm_reg_lambda'],
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif model_type == 'catboost':
        model = CatBoostClassifier(
            iterations=model_params['cb_iterations'],
            learning_rate=model_params['cb_learning_rate'],
            depth=model_params['cb_depth'],
            l2_leaf_reg=model_params['cb_l2_leaf_reg'],
            border_count=model_params['cb_border_count'],
            random_strength=model_params['cb_random_strength'],
            random_state=42,
            verbose=0,
            thread_count=-1
        )
    elif model_type == 'adaboost':
        base_estimator = DecisionTreeClassifier(
            max_depth=model_params['ada_base_max_depth'],
            random_state=42
        )

        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=model_params['ada_n_estimators'],
            learning_rate=model_params['ada_learning_rate'],
            random_state=42
        )

    if model is None:
        raise ValueError(f"Failed to reconstruct model type '{model_type}'")
    return model

def build_final_preprocessor(preprocessor_params, num_features, cat_features):
    """Build the final preprocessor pipeline using the best parameters from Optuna."""

    num_imputer_strategy = preprocessor_params['num_imputer_strategy']
    # Get KNN neighbors if KNN imputation was chosen
    knn_imputer_n_neighbors = preprocessor_params.get('knn_imputer_n_neighbors', None)


    cat_imputer_strategy = preprocessor_params['cat_imputer_strategy']
    cat_imputer_fill_value = 'missing' if cat_imputer_strategy == 'constant' else None

    scaler_type = preprocessor_params['scaler_type']
    if scaler_type == 'standard':
        scaler = StandardScaler()
    # Removed minmax check based on reduced options in tuning
    else: # robust
        scaler = RobustScaler()

    apply_log_transform = preprocessor_params['apply_log_transform']

    ohe_handle_unknown = preprocessor_params['ohe_handle_unknown']
    ohe_min_frequency = preprocessor_params.get('ohe_min_frequency', None)


    # Build numeric pipeline steps using the best parameters
    numeric_steps = []
    # 1. Imputation
    if num_imputer_strategy == 'knn':
        # Ensure knn_imputer_n_neighbors is present if strategy is 'knn'
        if knn_imputer_n_neighbors is None:
             raise ValueError("KNN imputer strategy chosen but knn_imputer_n_neighbors not found in params.")
        numeric_steps.append(('imputer', KNNImputer(n_neighbors=knn_imputer_n_neighbors)))
    else:
        numeric_steps.append(('imputer', SimpleImputer(strategy=num_imputer_strategy)))

    # 2. Log Transformation (Apply after imputation)
    if apply_log_transform:
         numeric_steps.append(('log_transform', FunctionTransformer(func=np.log1p, validate=False)))

    # 3. Scaling (Apply after imputation and transformation)
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

    X = engineer_features(X)

    # Determine features before split to ensure consistency
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure target variable has at least 2 classes for stratification
    stratify_option = None
    if len(y.unique()) >= 2:
        stratify_option = y
    else:
        print("Warning: Target variable does not have at least 2 classes. Stratification skipped.")

    # X = X.dropna(subset=['race'])  # hoặc fillna nếu phù hợp
    # y = y[X.index]

    # stratify_option = X['race']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42#, stratify=stratify_option
    )

    supported_models = [
        'knn', 'logistic', 'svc', 'decision_tree', 'random_forest',
        #'gradient_boosting', 'gaussian_nb', 'mlp',
        'xgboost', 'lightgbm', 'catboost', 'adaboost'
    ]
    all_model_results = {}
    studies = {} # Store study objects to potentially access best trial parameters

    print(f"\n--- Starting Optuna Hyperparameter Tuning ({n_trials_per_model} Trials Per Model) ---")

    for model_type in tqdm(supported_models, desc="Model Types Tuned"):
        print(f"\nTuning {model_type.upper()}...")
        # Use a unique storage name for each study if you were storing persistently
        # study_name = f"study_{model_type}"
        # study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), study_name=study_name, storage='sqlite:///optuna.db', load_if_exists=True)

        # For in-memory studies
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

        # Create a partial function with fixed arguments for the objective
        objective_func = lambda trial: objective_function(trial, X_train, y_train, X_val, y_val, num_features, cat_features, model_type)

        try:
            study.optimize(
                objective_func,
                n_trials=n_trials_per_model,
                n_jobs=-1, # Use all available CPU cores for trials
                show_progress_bar=False # tqdm handles outer loop, Optuna progress bar can be noisy
            )

            if study.best_trial:
                all_model_results[model_type] = {
                    'best_score': study.best_value,
                    'best_params': study.best_trial.params # Store the full best trial parameters
                }
                studies[model_type] = study # Store the study object
                print(f"  Best F1 Macro for {model_type.upper()}: {study.best_value:.4f}")
            else:
                 print(f"  No trials completed successfully for {model_type.upper()}.")

        except Exception as e:
            print(f"An error occurred during tuning for {model_type.upper()}: {e}")
            # Continue to the next model type


    print(f"\n--- Optuna Tuning Finished ---")

    if not all_model_results:
        print("Error: Optuna tuning failed for all model types. No models to ensemble. Exiting.")
        sys.exit(1)

    # Sort models by their best validation score
    sorted_models = sorted(all_model_results.items(), key=lambda item: item[1]['best_score'], reverse=True)

    # Determine how many top models to ensemble
    top_n = 3
    if len(sorted_models) < top_n:
        print(f"Warning: Fewer than {top_n} models were successfully tuned ({len(sorted_models)}). Using all available models.")
        top_n = len(sorted_models)

    if top_n == 0:
        print("Error: No models available to form an ensemble after tuning. Exiting.")
        sys.exit(1)

    top_models_info = sorted_models[:top_n]

    print("\n--- Top Performing Models ---")
    for i, (model_type, results) in enumerate(top_models_info):
        print(f"{i+1}. Model: {model_type.upper()}, Best Val F1 Macro: {results['best_score']:.4f}")

    # Use the preprocessor parameters from the single best trial among the top N models
    # The very first model in top_models_info is the model type whose best trial achieved the highest score
    best_tuning_trial_params = top_models_info[0][1]['best_params']

    # Extract only the preprocessor-specific parameters
    preprocessor_param_keys = [
        'num_imputer_strategy', 'knn_imputer_n_neighbors', 'apply_log_transform',
        'scaler_type', 'cat_imputer_strategy', 'ohe_handle_unknown', 'ohe_min_frequency'
    ]
    best_preprocessor_params = {k: best_tuning_trial_params.get(k, None) for k in preprocessor_param_keys}
    # Remove None values if a parameter wasn't applicable (e.g., knn_imputer_n_neighbors if knn wasn't chosen)
    best_preprocessor_params = {k: v for k, v in best_preprocessor_params.items() if v is not None}


    print(f"\nUsing preprocessor settings from the best single trial's parameters for the ensemble.")
    print("Best Preprocessor Parameters Selected:")
    for key, value in best_preprocessor_params.items():
        print(f"  {key}: {value}")


    # Build the final preprocessor using the selected parameters
    final_preprocessor = build_final_preprocessor(best_preprocessor_params, num_features, cat_features)


    # Reconstruct the top models using their respective best tuning parameters for the ensemble
    estimators = []
    print("\nReconstructing Top Models for Ensemble:")
    for model_type, results in top_models_info:
        # Get the best trial parameters for this specific model type
        model_specific_best_params = results['best_params']

        # Extract *only* model specific params, excluding preprocessor ones
        # Ensure we exclude *all* preprocessor keys defined earlier
        model_params_only = {k: v for k, v in model_specific_best_params.items() if k not in preprocessor_param_keys}

        print(f"  - {model_type.upper()}")
        # Optionally print model params here if desired
        # print(f"    Model Params: {model_params_only}")

        model_instance = reconstruct_model(model_type, model_params_only)
        estimators.append((model_type, model_instance))

    print("\n--- Evaluating Voting Classifiers on Validation Set ---")

    # Hard Voting
    voting_clf_hard = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    pipeline_hard = Pipeline([('preprocessor', final_preprocessor), ('classifier', voting_clf_hard)])
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


    print(f"\n--- Finalizing Best Voting Model ({best_voting_type.upper()}) ---")
    # The best_pipeline is already fitted on X_train, y_train with the chosen voting type

    y_pred_final_val = best_pipeline.predict(X_val)
    final_accuracy = accuracy_score(y_val, y_pred_final_val)
    print(f"Final Validation Accuracy ({best_voting_type.upper()} Voting): {final_accuracy:.4f}")
    print(f"Final Validation F1 Macro ({best_voting_type.upper()} Voting): {best_voting_f1:.4f}")

    print(f"\n--- Final Model Configuration ({best_voting_type.upper()} Voting) ---")

    print("\nPreprocessor Parameters:")
    for key, value in best_preprocessor_params.items():
        print(f"  {key}: {value}")

    print("\nBase Estimators and their Tuned Parameters:")
    # top_models_info contains (model_type, results_dict) for the top N models
    for model_type, results in top_models_info:
        print(f"  - {model_type.upper()}")
        # Extract only the model-specific parameters from the best trial for this model type
        model_params = {k: v for k, v in results['best_params'].items() if k not in preprocessor_param_keys}
        for param_name, param_value in model_params.items():
            print(f"    {param_name}: {param_value}")


    # Save components
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
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

    model_files = [f for f in os.listdir(model_dir) if f.startswith('trained_') and f.endswith('_voting_model.joblib')]
    model_path = os.path.join(model_dir, model_files[0])
    print(f"Using model file: {model_path}")


    if not os.path.exists(preprocessor_path):
        print(f"Error: Preprocessor file not found at {preprocessor_path}. Exiting.")
        sys.exit(1)
    loaded_preprocessor = load(preprocessor_path)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Exiting.")
        sys.exit(1)
    loaded_model = load(model_path)

    test_path = os.path.join(test_input_dir, 'test.json')
    
    if not os.path.exists(test_path):
        print(f"Error: Test data file not found at {test_path}. Exiting.")
        sys.exit(1)
    df_test = pd.read_json(test_path, lines=True)

    df_test = engineer_features(df_test)
    
    print(f"Loaded {len(df_test)} test cases.")
    if df_test.empty:
       print("Warning: Test data is empty.")
       pd.DataFrame({'two_year_recid': []}).to_json(output_path, orient='records', lines=True)
       print("Empty prediction file created.")
       print("--- Prediction Finished ---")
       sys.exit(0)

    X_test_processed = loaded_preprocessor.transform(df_test)
    predictions = loaded_model.predict(X_test_processed)
    print(f"Generated {len(predictions)} predictions.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({'two_year_recid': predictions}).to_json(output_path, orient='records', lines=True)
    print(f"Predictions saved to: {output_path}")

    print("--- Prediction Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Train/Predict Recidivism Voting Ensemble Model")
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    parser_train = subparsers.add_parser('train', help='Tune models, create best voting ensemble, save preprocessor and model')
    parser_train.add_argument('--public_dir', type=str, default='./', help='Directory containing train_data/train.json')
    parser_train.add_argument('--model_dir', type=str, default ='./model_output', help='Directory to save preprocessor.joblib and trained_voting_model.joblib') # Changed default output dir
    parser_train.add_argument('--n_trials_per_model', type=int, default=30, help='Number of Optuna trials FOR EACH base model type (default: 15)')

    parser_predict = subparsers.add_parser('predict', help='Make predictions using saved preprocessor and voting model')
    parser_predict.add_argument('--model_dir', type=str, default='./model_output', help='Directory containing preprocessor.joblib and trained_..._voting_model.joblib') # Changed default input dir
    parser_predict.add_argument('--test_input_dir', type=str, default='./test_data', help='Directory containing test.json')
    parser_predict.add_argument('--output_path', type=str, default='./predict.json', help='File path for saving predictions')

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args.public_dir, args.model_dir, args.n_trials_per_model)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.test_input_dir, args.output_path)

if __name__ == "__main__":
    main()