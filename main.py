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


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from joblib import dump, load
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    model = None
    params = {}

    if model_type == 'knn':
        params['knn_n_neighbors'] = trial.suggest_int('knn_n_neighbors', 3, 15) # Reduced range
        params['knn_weights'] = trial.suggest_categorical('knn_weights', ['uniform', 'distance'])
        params['knn_metric'] = trial.suggest_categorical('knn_metric', ['minkowski', 'manhattan']) # Reduced options
        model = KNeighborsClassifier(n_neighbors=params['knn_n_neighbors'], weights=params['knn_weights'], metric=params['knn_metric'], n_jobs=-1)

    elif model_type == 'logistic':
        params['logreg_c'] = trial.suggest_float("logreg_c", 1e-2, 1e1, log=True) # Reduced range
        params['logreg_penalty'] = trial.suggest_categorical('logreg_penalty', ['l1', 'l2'])
        solver = 'liblinear'
        model = LogisticRegression(C=params['logreg_c'], penalty=params['logreg_penalty'], solver=solver, max_iter=1000, random_state=42, n_jobs=-1)

    elif model_type == 'svc':
        params['svc_c'] = trial.suggest_float("svc_c", 1e-1, 1e2, log=True) # Reduced range
        params['svc_kernel'] = trial.suggest_categorical("svc_kernel", ["linear", "rbf"]) # Reduced options
        gamma = "scale"
        degree = 3

        if params['svc_kernel'] == "rbf": # Only tune gamma for rbf
            params['svc_gamma_type'] = trial.suggest_categorical("svc_gamma_type", ["scale", "float"]) # Reduced options
            if params['svc_gamma_type'] == "float":
                params['svc_gamma_float'] = trial.suggest_float("svc_gamma_float", 1e-3, 1.0, log=True) # Reduced range
                gamma = params['svc_gamma_float']
            else:
                gamma = params['svc_gamma_type'] # scale

        # Degree tuning removed as 'poly' kernel is removed

        params['_svc_calculated_gamma'] = gamma
        params['_svc_calculated_degree'] = degree
        model = SVC(C=params['svc_c'], kernel=params['svc_kernel'], gamma=gamma, degree=degree, probability=True, random_state=42)

    elif model_type == 'decision_tree':
        params['dt_criterion'] = trial.suggest_categorical("dt_criterion", ["gini", "entropy"])
        params['dt_max_depth'] = trial.suggest_int("dt_max_depth", 5, 25) # Reduced range
        params['dt_min_samples_split'] = trial.suggest_int("dt_min_samples_split", 2, 10) # Reduced range
        params['dt_min_samples_leaf'] = trial.suggest_int("dt_min_samples_leaf", 1, 10) # Reduced range
        model = DecisionTreeClassifier(
            criterion=params['dt_criterion'],
            max_depth=params['dt_max_depth'],
            min_samples_split=params['dt_min_samples_split'],
            min_samples_leaf=params['dt_min_samples_leaf'],
            random_state=42
        )

    elif model_type == 'random_forest':
        params['rf_n_estimators'] = trial.suggest_int("rf_n_estimators", 50, 200) # Reduced range
        params['rf_max_depth_type'] = trial.suggest_categorical("rf_max_depth_type", ["int", "none"])
        max_depth = None
        if params['rf_max_depth_type'] == "int":
             params['rf_max_depth_int'] = trial.suggest_int("rf_max_depth_int", 10, 30) # Reduced range
             max_depth = params['rf_max_depth_int']
        else:
             max_depth = None

        params['rf_min_samples_split'] = trial.suggest_int("rf_min_samples_split", 2, 10) # Reduced range
        params['rf_min_samples_leaf'] = trial.suggest_int("rf_min_samples_leaf", 1, 10) # Reduced range
        params['rf_max_features_type'] = trial.suggest_categorical("rf_max_features_type", ["sqrt", "log2"]) # Reduced options
        max_features = params['rf_max_features_type']
        # Float option removed
        # None option removed

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
        params['gb_n_estimators'] = trial.suggest_int("gb_n_estimators", 50, 200) # Reduced range
        params['gb_learning_rate'] = trial.suggest_float("gb_learning_rate", 1e-2, 0.3, log=True) # Reduced range
        params['gb_max_depth'] = trial.suggest_int("gb_max_depth", 3, 7) # Reduced range
        params['gb_subsample'] = trial.suggest_float("gb_subsample", 0.7, 1.0) # Reduced range
        params['gb_min_samples_split'] = trial.suggest_int("gb_min_samples_split", 2, 10) # Reduced range
        params['gb_min_samples_leaf'] = trial.suggest_int("gb_min_samples_leaf", 1, 10) # Reduced range
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
        params['gnb_var_smoothing'] = trial.suggest_float("gnb_var_smoothing", 1e-9, 1e-5, log=True) # Reduced range
        model = GaussianNB(var_smoothing=params['gnb_var_smoothing'])

    elif model_type == 'mlp':
        params['mlp_hidden_layer_neurons'] = trial.suggest_int("mlp_hidden_layer_neurons", 50, 150, log=True) # Reduced range
        hidden_layer_sizes = (params['mlp_hidden_layer_neurons'],)
        params['mlp_activation'] = trial.suggest_categorical("mlp_activation", ["relu", "tanh"]) # Reduced options
        mlp_solver = 'adam'
        params['mlp_alpha'] = trial.suggest_float("mlp_alpha", 1e-5, 1e-3, log=True) # Reduced range
        params['mlp_learning_rate_init'] = trial.suggest_float("mlp_learning_rate_init", 1e-3, 1e-2, log=True) # Reduced range
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=params['mlp_activation'],
            solver=mlp_solver,
            alpha=params['mlp_alpha'],
            learning_rate_init=params['mlp_learning_rate_init'],
            max_iter=1000,
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
        model = xgb.XGBClassifier(
            n_estimators=params['xgb_n_estimators'],
            learning_rate=params['xgb_learning_rate'],
            max_depth=params['xgb_max_depth'],
            subsample=params['xgb_subsample'],
            colsample_bytree=params['xgb_colsample_bytree'],
            gamma=params['xgb_gamma'],
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
        model = lgb.LGBMClassifier(
            n_estimators=params['lgbm_n_estimators'],
            learning_rate=params['lgbm_learning_rate'],
            num_leaves=params['lgbm_num_leaves'],
            max_depth=params['lgbm_max_depth'],
            subsample=params['lgbm_subsample'],
            colsample_bytree=params['lgbm_colsample_bytree'],
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
        model = CatBoostClassifier(
            iterations=params['cb_iterations'],
            learning_rate=params['cb_learning_rate'],
            depth=params['cb_depth'],
            l2_leaf_reg=params['cb_l2_leaf_reg'],
            border_count=params['cb_border_count'],
            random_state=42,
            verbose=0,
            thread_count=-1
        )


    elif model_type == 'adaboost':
        params['ada_n_estimators'] = trial.suggest_int("ada_n_estimators", 50, 200)
        params['ada_learning_rate'] = trial.suggest_float("ada_learning_rate", 0.01, 1.0, log=True)
        params['ada_base_max_depth'] = trial.suggest_int("ada_base_max_depth", 1, 4)
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=params['ada_base_max_depth'], random_state=42),
            n_estimators=params['ada_n_estimators'],
            learning_rate=params['ada_learning_rate'],
            random_state=42
        )


    return model, params


def objective_function(trial, X_train, y_train, X_val, y_val, num_features, cat_features, model_type):
    num_imputer_strategy = trial.suggest_categorical('num_imputer_strategy', ['mean', 'median']) # Reduced options
    cat_imputer_strategy = trial.suggest_categorical('cat_imputer_strategy', ['most_frequent', 'constant'])
    cat_imputer_fill_value = 'missing' if cat_imputer_strategy == 'constant' else None
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'robust']) # Reduced options
    ohe_handle_unknown = trial.suggest_categorical('ohe_handle_unknown', ['ignore', 'infrequent_if_exist'])
    ohe_min_frequency = trial.suggest_float('ohe_min_frequency', 0.01, 0.05) if ohe_handle_unknown == 'infrequent_if_exist' else None # Reduced range


    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=num_imputer_strategy)),
        #('scaler', scaler)
    ])

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

    model, _ = get_model_and_params(trial, model_type)

    full_pipeline = Pipeline(steps=[('preprocessor', trial_preprocessor), ('classifier', model)])
    full_pipeline.fit(X_train, y_train)
    y_pred_val = full_pipeline.predict(X_val)
    score = f1_score(y_val, y_pred_val, average='macro')

    trial.report(score, step=0)
    if trial.should_prune():
            raise optuna.TrialPruned()

    return score

def reconstruct_model(model_type, model_params):
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
         model = MLPClassifier(
            hidden_layer_sizes=(model_params['mlp_hidden_layer_neurons'],),
            activation=model_params['mlp_activation'],
            solver='adam',
            alpha=model_params['mlp_alpha'],
            learning_rate_init=model_params['mlp_learning_rate_init'],
            max_iter=1500,
            early_stopping=False, # Keep False for final model as per original code
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
    num_imputer_strategy = preprocessor_params['num_imputer_strategy']
    cat_imputer_strategy = preprocessor_params['cat_imputer_strategy']
    cat_imputer_fill_value = 'missing' if cat_imputer_strategy == 'constant' else None
    scaler_type = preprocessor_params['scaler_type']
    ohe_handle_unknown = preprocessor_params['ohe_handle_unknown']
    ohe_min_frequency = preprocessor_params.get('ohe_min_frequency', None)

    if scaler_type == 'standard':
        scaler = StandardScaler()
    # Removed minmax check
    else: # robust
        scaler = RobustScaler()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=num_imputer_strategy)),
        ('scaler', scaler)
    ])

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


def run_train(public_dir, model_dir, n_trials_per_model=10):
    print("--- Starting Training ---")
    os.makedirs(model_dir, exist_ok=True)

    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    df = pd.read_json(train_path, lines=True)
    print(f"Loaded {len(df)} training cases.")

    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    # X = X.dropna(subset=['race'])  # hoặc fillna nếu phù hợp
    # y = y[X.index]

    # stratify_option = X['race']
    # test_size = 0.25

    X = engineer_features(X)

    stratify_option = None
    test_size = 0.25
    if len(y.unique()) >= 2:
        stratify_option = y


    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_option
    )

    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

    supported_models = [
        'knn', 'logistic', 'svc', 'decision_tree', 'random_forest',
        'gradient_boosting', 'gaussian_nb', 'mlp',
        'xgboost', 'lightgbm', 'catboost', 'adaboost'
    ]
    all_model_results = {}

    print(f"\n--- Starting Optuna Hyperparameter Tuning ({n_trials_per_model} Trials Per Model) ---")

    for model_type in tqdm(supported_models, desc="Model Types Tuned"):
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

        objective_func = lambda trial: objective_function(trial, X_train, y_train, X_val, y_val, num_features, cat_features, model_type)

        study.optimize(
            objective_func,
            n_trials=n_trials_per_model,
            n_jobs=-1,
        )

        if study.best_trial:
            all_model_results[model_type] = {
                'best_score': study.best_value,
                'best_params': study.best_params
            }


    print(f"\n--- Optuna Tuning Finished ---")

    if not all_model_results:
        print("Error: Optuna tuning failed for all model types. Exiting.")
        sys.exit(1)

    sorted_models = sorted(all_model_results.items(), key=lambda item: item[1]['best_score'], reverse=True)

    top_n = 3
    if len(sorted_models) < top_n:
        print(f"Warning: Fewer than {top_n} models were successfully tuned. Using the top {len(sorted_models)} models.")
        top_n = len(sorted_models)

    if top_n == 0:
        print("Error: No models available to form an ensemble. Exiting.")
        sys.exit(1)

    top_models_info = sorted_models[:top_n]

    print("\n--- Top Performing Models ---")
    for i, (model_type, results) in enumerate(top_models_info):
        print(f"{i+1}. Model: {model_type.upper()}, Best Val F1 Macro: {results['best_score']:.4f}")

    best_single_model_type = top_models_info[0][0]
    best_single_model_results = top_models_info[0][1]
    best_preprocessor_params = {k: v for k, v in best_single_model_results['best_params'].items() if k.startswith(('num_', 'cat_', 'scaler_', 'ohe_'))}
    print(f"\nUsing preprocessor settings from the best single model ({best_single_model_type.upper()}) for the ensemble.")
    print("Best Preprocessor Parameters:")
    print(best_preprocessor_params)

    final_preprocessor = build_final_preprocessor(best_preprocessor_params, num_features, cat_features)

    estimators = []
    print("\nReconstructing Top Models for Ensemble:")
    for model_type, results in top_models_info:
        model_params = {k: v for k, v in results['best_params'].items() if not k.startswith(('num_', 'cat_', 'scaler_', 'ohe_'))}
        print(f"  - {model_type.upper()}")
        model_instance = reconstruct_model(model_type, model_params)
        estimators.append((model_type, model_instance))

    print("\n--- Evaluating Voting Classifiers on Validation Set ---")

    voting_clf_hard = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    pipeline_hard = Pipeline([('preprocessor', final_preprocessor), ('classifier', voting_clf_hard)])
    pipeline_hard.fit(X_train, y_train)
    y_pred_val_hard = pipeline_hard.predict(X_val)
    f1_hard = f1_score(y_val, y_pred_val_hard, average='macro')
    print(f"Hard Voting F1 Macro: {f1_hard:.4f}")

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


    print(f"\n--- Finalizing Best Voting Model ({best_voting_type.upper()}) ---")
    fitted_preprocessor = best_pipeline.named_steps['preprocessor']
    trained_voting_model = best_pipeline.named_steps['classifier']

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
        # Get only the model-specific parameters, excluding preprocessor ones
        model_params = {k: v for k, v in results['best_params'].items() if not k.startswith(('num_', 'cat_', 'scaler_', 'ohe_'))}
        for param_name, param_value in model_params.items():
            print(f"    {param_name}: {param_value}")

    # Save components
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    model_path = os.path.join(model_dir, f'trained_{best_voting_type}_voting_model.joblib')

    print(f"\nSaving fitted preprocessor to: {preprocessor_path}")
    dump(fitted_preprocessor, preprocessor_path)

    print(f"Saving trained {best_voting_type.upper()} Voting Classifier model to: {model_path}")
    dump(trained_voting_model, model_path)

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
    parser_train.add_argument('--model_dir', type=str, default ='./', help='Directory to save preprocessor.joblib and trained_voting_model.joblib')
    parser_train.add_argument('--n_trials_per_model', type=int, default=15, help='Number of Optuna trials FOR EACH base model type (default: 15)') # Reduced default

    parser_predict = subparsers.add_parser('predict', help='Make predictions using saved preprocessor and voting model')
    parser_predict.add_argument('--model_dir', type=str, default='./', help='Directory containing preprocessor.joblib and trained_..._voting_model.joblib')
    parser_predict.add_argument('--test_input_dir', type=str, default='./test_data', help='Directory containing test.json')
    parser_predict.add_argument('--output_path', type=str, default='./predict.json', help='File path for saving predictions')

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args.public_dir, args.model_dir, args.n_trials_per_model)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.test_input_dir, args.output_path)

if __name__ == "__main__":
    main()