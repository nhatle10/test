import argparse
import sys
import os
import pandas as pd
import numpy as np
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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from joblib import dump, load
from tqdm import tqdm

# Suppress specific warnings, including the one about n_jobs with liblinear
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.linear_model._logistic") # Specifically target logistic regression warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network._multilayer_perceptron") # Specifically target MLPConvergenceWarning

# Optuna specific warning suppression is removed as Optuna is not used

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
    # Điền giá trị NaN bằng 0 cho các cột vi phạm vị thành niên
    df[['juv_fel_count', 'juv_misd_count', 'juv_other_count']] = df[[
        'juv_fel_count', 'juv_misd_count', 'juv_other_count'
    ]].fillna(0)

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

    #df.drop(columns=['juv_fel_count', 'juv_misd_count', 'juv_other_count'], inplace=True)

    return df

def get_pipeline_and_param_grid(model_type, num_features, cat_features):
    # Define preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()) # Placeholder scaler, type is tuned
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('onehot', OneHotEncoder(sparse_output=False)) # handle_unknown, min_frequency are tuned
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough',
        n_jobs=-1
    )

    # Define model-specific pipeline and parameter grid
    model = None
    model_param_grid = {}

    if model_type == 'knn':
        model = KNeighborsClassifier(n_jobs=-1)
        model_param_grid = {
            'classifier__n_neighbors': [5, 9, 13], # Example values
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['minkowski', 'manhattan'],
        }

    elif model_type == 'logistic':
        # Set n_jobs=1 explicitly for liblinear to suppress warning, or just omit as default is 1
        model = LogisticRegression(solver='liblinear', max_iter=2000, random_state=42)
        model_param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10], # Example values
            'classifier__penalty': ['l1', 'l2'],
        }

    elif model_type == 'svc':
        model = SVC(probability=True, random_state=42)
        model_param_grid = {
            'classifier__C': [0.1, 1, 10], # Example values
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 0.01, 0.1], # Example values
        }

    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
        model_param_grid = {
            'classifier__criterion': ["gini", "entropy"],
            'classifier__max_depth': [10, 15, 20], # Example values
            'classifier__min_samples_split': [2, 5, 10], # Example values
            'classifier__min_samples_leaf': [1, 5, 10], # Example values
        }

    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        model_param_grid = {
            'classifier__n_estimators': [100, 150, 200], # Example values
            'classifier__max_depth': [15, 25, None], # Example values
            'classifier__min_samples_split': [2, 5], # Example values
            'classifier__min_samples_leaf': [1, 5], # Example values
            'classifier__max_features': ['sqrt', 'log2'], # Example values
            'classifier__criterion': ["gini", "entropy"],
        }

    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
        model_param_grid = {
            'classifier__n_estimators': [100, 150, 200], # Example values
            'classifier__learning_rate': [0.05, 0.1, 0.2], # Example values
            'classifier__max_depth': [3, 5, 7], # Example values
            'classifier__subsample': [0.8, 0.9, 1.0], # Example values
            'classifier__min_samples_split': [2, 5], # Example values
            'classifier__min_samples_leaf': [1, 5], # Example values
        }

    elif model_type == 'gaussian_nb':
        model = GaussianNB()
        model_param_grid = {
            'classifier__var_smoothing': [1e-9, 1e-7, 1e-5], # Example values
        }

    elif model_type == 'mlp':
        # Max_iter increased and early_stopping added for training stability
        model = MLPClassifier(max_iter=1500, early_stopping=True, n_iter_no_change=10, random_state=42)
        model_param_grid = {
            'classifier__hidden_layer_sizes': [(100,), (125,), (150,)], # Example values
            'classifier__activation': ["relu", "tanh"], # Example values
            'classifier__alpha': [1e-5, 1e-4, 1e-3], # Example values
            'classifier__learning_rate_init': [1e-3, 1e-2], # Example values
            'classifier__solver': ['adam'] # Fixed solver as per original
        }

    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        model_param_grid = {
            'classifier__n_estimators': [100, 200, 300], # Example values
            'classifier__learning_rate': [0.05, 0.1, 0.15], # Example values
            'classifier__max_depth': [3, 5, 7], # Example values
            'classifier__subsample': [0.7, 0.8, 0.9], # Example values
            'classifier__colsample_bytree': [0.7, 0.8, 0.9], # Example values
            'classifier__gamma': [0, 0.1, 0.3], # Example values
        }

    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        model_param_grid = {
            'classifier__n_estimators': [100, 200, 300], # Example values
            'classifier__learning_rate': [0.05, 0.1, 0.15], # Example values
            'classifier__num_leaves': [25, 35, 45], # Example values
            'classifier__max_depth': [4, 6], # Example values
            'classifier__subsample': [0.7, 0.8, 0.9], # Example values
            'classifier__colsample_bytree': [0.7, 0.8, 0.9], # Example values
        }

    elif model_type == 'catboost':
        model = CatBoostClassifier(random_state=42, verbose=0, thread_count=-1)
        model_param_grid = {
            'classifier__iterations': [100, 200, 300], # Example values
            'classifier__learning_rate': [0.05, 0.1, 0.15], # Example values
            'classifier__depth': [4, 6], # Example values
            'classifier__l2_leaf_reg': [3, 5, 8], # Example values
            'classifier__border_count': [64, 128], # Example values
        }

    elif model_type == 'adaboost':
        # Base estimator depth is tuned within AdaBoost
        base_estimator = DecisionTreeClassifier(random_state=42)
        model = AdaBoostClassifier(estimator=base_estimator, random_state=42)
        model_param_grid = {
            'classifier__n_estimators': [100, 150, 200], # Example values
            'classifier__learning_rate': [0.1, 0.5, 1.0], # Example values
            'classifier__estimator__max_depth': [1, 2, 3], # Example values for base DT depth
        }
    else:
         raise ValueError(f"Unsupported model type: {model_type}")


    # Define preprocessing parameter grid (common across models, tuned for each)
    preprocessor_param_grid = {
        'preprocessor__num__imputer__strategy': ['mean', 'median'], # Example values
        'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'], # Example values
        'preprocessor__cat__imputer__fill_value': [None, 'missing'], # Value to use if strategy is 'constant'
        'preprocessor__num__scaler': [StandardScaler(), RobustScaler()], # Example values
        'preprocessor__cat__onehot__handle_unknown': ['ignore', 'infrequent_if_exist'], # Example values
        'preprocessor__cat__onehot__min_frequency': [0.01, 0.05, 0.1], # Example values, used if handle_unknown='infrequent_if_exist'
    }
    # Note: The combination of cat__imputer__strategy='constant' and cat__imputer__fill_value=None will be invalid.
    # GridSearchCV handles this by failing the fit for that specific combination.

    # Combine preprocessor and model grids
    full_param_grid = {**preprocessor_param_grid, **model_param_grid}

    # Create the pipeline with named steps
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline, full_param_grid


def build_final_preprocessor(best_preprocessor_params, num_features, cat_features):
    # Extract params specifically for preprocessor steps
    # Need to map generic names like 'scaler' back to actual scaler instances
    best_scaler_name = best_preprocessor_params.get('preprocessor__num__scaler', StandardScaler()) # Default

    # Determine the correct scaler instance from the grid search result
    # The value in best_preprocessor_params['preprocessor__num__scaler'] will be the actual instance
    scaler_instance = best_scaler_name

    num_imputer_strategy = best_preprocessor_params.get('preprocessor__num__imputer__strategy', 'mean')
    cat_imputer_strategy = best_preprocessor_params.get('preprocessor__cat__imputer__strategy', 'most_frequent')
    cat_imputer_fill_value = best_preprocessor_params.get('preprocessor__cat__imputer__fill_value', None)
    ohe_handle_unknown = best_preprocessor_params.get('preprocessor__cat__onehot__handle_unknown', 'ignore')
    ohe_min_frequency = best_preprocessor_params.get('preprocessor__cat__onehot__min_frequency', None)

    # Handle the case where min_frequency is None when handle_unknown='ignore'
    # The grid search might still explore this, but the OHE constructor handles it.
    # Ensure fill_value is set correctly if strategy is constant
    if cat_imputer_strategy == 'constant' and cat_imputer_fill_value is None:
         # This combination would likely fail or behave unexpectedly in grid search,
         # but explicitly handle it here just in case the best params include it.
         # In practice, the grid should probably exclude this invalid combination.
         # For this code, we'll assume grid search didn't return this invalid state
         # or that SimpleImputer handles strategy='constant' with fill_value=None (it defaults to 0 or 'missing')
         pass # SimpleImputer with strategy='constant' requires fill_value

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=num_imputer_strategy)),
        ('scaler', scaler_instance) # Use the best scaler instance
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


def reconstruct_model_from_params(model_type, model_params):
    # This function reconstructs a model instance using the parameters
    # found by GridSearchCV.
    model = None

    # Remove 'classifier__' prefix from parameter names
    cleaned_params = {k.replace('classifier__', ''): v for k, v in model_params.items()}

    if model_type == 'knn':
        model = KNeighborsClassifier(**cleaned_params, n_jobs=-1)
    elif model_type == 'logistic':
        # Ensure n_jobs is not included in cleaned_params for liblinear solver
        logistic_params = cleaned_params.copy()
        # The n_jobs parameter is not in the param_grid for logistic regression with liblinear,
        # so it won't be in cleaned_params anyway.
        model = LogisticRegression(**logistic_params, solver='liblinear', max_iter=2000, random_state=42) # n_jobs defaults to 1
    elif model_type == 'svc':
        model = SVC(**cleaned_params, probability=True, random_state=42)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(**cleaned_params, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**cleaned_params, random_state=42, n_jobs=-1)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(**cleaned_params, random_state=42)
    elif model_type == 'gaussian_nb':
        model = GaussianNB(**cleaned_params)
    elif model_type == 'mlp':
        # MLP hidden_layer_sizes needs careful handling if it was tuned as (size,) tuple
        # Ensure max_iter is sufficient and early_stopping is False for the final model as intended
        mlp_params = cleaned_params.copy()
        if 'hidden_layer_sizes' in mlp_params and isinstance(mlp_params['hidden_layer_sizes'], int):
             mlp_params['hidden_layer_sizes'] = (mlp_params['hidden_layer_sizes'],)
        model = MLPClassifier(**mlp_params, max_iter=1500, early_stopping=False, random_state=42) # early_stopping=False for final fit
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(**cleaned_params, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(**cleaned_params, random_state=42, n_jobs=-1, verbose=-1)
    elif model_type == 'catboost':
        model = CatBoostClassifier(**cleaned_params, random_state=42, verbose=0, thread_count=-1)
    elif model_type == 'adaboost':
        # AdaBoost base estimator needs to be reconstructed separately
        base_estimator_params = {k.replace('estimator__', ''): v for k, v in cleaned_params.items() if k.startswith('estimator__')}
        base_estimator = DecisionTreeClassifier(**base_estimator_params, random_state=42)
        # Remove estimator params from the main cleaned_params for AdaBoost itself
        ada_params = {k: v for k, v in cleaned_params.items() if not k.startswith('estimator__')}
        model = AdaBoostClassifier(estimator=base_estimator, **ada_params, random_state=42)

    if model is None:
        raise ValueError(f"Failed to reconstruct model type '{model_type}'")
    return model

def run_train(public_dir, model_dir, n_cv_folds=5):
    print("--- Starting Training (Grid Search) ---")
    os.makedirs(model_dir, exist_ok=True)

    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    df = pd.read_json(train_path, lines=True)
    print(f"Loaded {len(df)} training cases.")

    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    # Engineer features
    X = engineer_features(X)

    # Define features after engineering
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Split data into training and validation set for grid search evaluation
    # We use this split for GridSearchCV with cross-validation *on the train split*
    # and then evaluate the final ensemble on the validation split.
    # The final ensemble model will be trained on the entire X, y.
    stratify_option = None
    test_size = 0.25
    if len(y.unique()) >= 2:
        stratify_option = y

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_option
    )
    print(f"Split data into {len(X_train)} training and {len(X_val)} validation samples.")


    supported_models = [
        'knn', 'logistic', 'svc', 'decision_tree', 'random_forest',
        'gradient_boosting', 'gaussian_nb', 'mlp',
        'xgboost', 'lightgbm', 'catboost', 'adaboost'
    ]
    all_model_results = {}
    f1_macro_scorer = make_scorer(f1_score, average='macro')

    print(f"\n--- Starting Grid Search Hyperparameter Tuning ({n_cv_folds}-Fold CV on Training Data) ---")

    for model_type in tqdm(supported_models, desc="Model Types Tuned"):
        try:
            pipeline, param_grid = get_pipeline_and_param_grid(model_type, num_features, cat_features)

            # Use Stratified K-Fold cross-validation for Grid Search on the training data
            cv_strategy = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=f1_macro_scorer, # Use macro F1 score
                cv=cv_strategy,
                n_jobs=-1,
                verbose=0 # Suppress verbose output
            )

            print(f"  - Tuning {model_type.upper()}...")
            grid_search.fit(X_train, y_train)

            all_model_results[model_type] = {
                'best_score': grid_search.best_score_, # Mean CV score
                'best_params': grid_search.best_params_
            }
            print(f"    Best CV Score: {grid_search.best_score_:.4f}")
            # print(f"    Best Params: {grid_search.best_params_}") # Optional: print params during tuning

        except Exception as e:
            print(f"Error tuning {model_type.upper()}: {e}")
            all_model_results[model_type] = {'best_score': -1, 'best_params': None} # Indicate failure


    print(f"\n--- Grid Search Tuning Finished ---")

    successful_models = {k: v for k, v in all_model_results.items() if v['best_score'] >= 0}

    if not successful_models:
        print("Error: Grid Search tuning failed for all model types. Exiting.")
        sys.exit(1)

    sorted_models = sorted(successful_models.items(), key=lambda item: item[1]['best_score'], reverse=True)

    top_n = 3
    if len(sorted_models) < top_n:
        print(f"Warning: Fewer than {top_n} models were successfully tuned. Using the top {len(sorted_models)} models.")
        top_n = len(sorted_models)

    if top_n == 0:
        print("Error: No successful models available to form an ensemble. Exiting.")
        sys.exit(1)

    top_models_info = sorted_models[:top_n]

    print("\n--- Top Performing Models (Based on Mean CV F1 Macro) ---")
    for i, (model_type, results) in enumerate(top_models_info):
        print(f"{i+1}. Model: {model_type.upper()}, Mean CV F1 Macro: {results['best_score']:.4f}")
        # print(f"   Best Params: {results['best_params']}") # Optional: print best params found

    # Use preprocessor settings from the overall best performing model (top 1)
    best_single_model_type = top_models_info[0][0]
    best_single_model_results = top_models_info[0][1]

    # Extract only preprocessor parameters from the best params of the top model
    best_preprocessor_params = {k: v for k, v in best_single_model_results['best_params'].items() if k.startswith('preprocessor__')}

    print(f"\nUsing preprocessor settings from the best single model ({best_single_model_type.upper()}) for the ensemble.")
    # print("Best Preprocessor Parameters:")
    # print(best_preprocessor_params) # Optional: print best preprocessor params

    # Build the final preprocessor using the selected best parameters
    final_preprocessor = build_final_preprocessor(best_preprocessor_params, num_features, cat_features)


    print("\nReconstructing Top Models with Best Params for Ensemble:")
    estimators = []
    # Filter out preprocessor params before passing to reconstruct_model_from_params
    for model_type, results in top_models_info:
        model_params_only = {k: v for k, v in results['best_params'].items() if not k.startswith('preprocessor__')}
        print(f"  - {model_type.upper()}")
        model_instance = reconstruct_model_from_params(model_type, model_params_only)
        estimators.append((model_type, model_instance))

    # Check if soft voting is possible (all estimators must support predict_proba)
    supports_predict_proba = all(hasattr(model, 'predict_proba') for _, model in estimators)

    print("\n--- Evaluating Voting Classifiers on Validation Set ---")

    # Hard Voting
    voting_clf_hard = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    # Build the final pipeline with the chosen preprocessor and voting classifier
    pipeline_hard = Pipeline([('preprocessor', final_preprocessor), ('classifier', voting_clf_hard)])
    # Train the pipeline on X_train, y_train for validation evaluation
    print("Fitting Hard Voting Pipeline on training data for validation check...")
    pipeline_hard.fit(X_train, y_train)
    y_pred_val_hard = pipeline_hard.predict(X_val)
    f1_hard = f1_score(y_val, y_pred_val_hard, average='macro')
    acc_hard = accuracy_score(y_val, y_pred_val_hard)
    print(f"Hard Voting Validation F1 Macro: {f1_hard:.4f}, Accuracy: {acc_hard:.4f}")

    best_pipeline = pipeline_hard
    best_voting_type = 'hard'
    best_voting_f1 = f1_hard
    best_voting_acc = acc_hard

    # Soft Voting (only if all models support predict_proba)
    if supports_predict_proba:
        voting_clf_soft = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        pipeline_soft = Pipeline([('preprocessor', final_preprocessor), ('classifier', voting_clf_soft)])
        print("\nFitting Soft Voting Pipeline on training data for validation check...")
        pipeline_soft.fit(X_train, y_train)
        y_pred_val_soft = pipeline_soft.predict(X_val)
        f1_soft = f1_score(y_val, y_pred_val_soft, average='macro')
        acc_soft = accuracy_score(y_val, y_pred_val_soft)
        print(f"Soft Voting Validation F1 Macro: {f1_soft:.4f}, Accuracy: {acc_soft:.4f}")

        if f1_soft > f1_hard:
            best_pipeline = pipeline_soft
            best_voting_type = 'soft'
            best_voting_f1 = f1_soft
            best_voting_acc = acc_soft
            print("Soft voting performed better on validation set (based on F1 Macro).")
        else:
             print("Hard voting performed better or equal on validation set (based on F1 Macro).")
    else:
        print("\nSoft voting is not possible because not all selected base models support predict_proba.")


    print(f"\n--- Finalizing Best Voting Model ({best_voting_type.upper()}) ---")

    # Retrain the best pipeline on the *entire* dataset (X, y)
    # This is standard practice after parameter tuning on a subset or with CV.
    # It ensures the final model learns from all available training data.
    print(f"Training final {best_voting_type.upper()} Voting Pipeline on the full training dataset...")

    # Reconstruct the estimators again to ensure they are fresh instances for the final fit
    final_estimators = []
    for model_type, results in top_models_info:
        model_params_only = {k: v for k, v in results['best_params'].items() if not k.startswith('preprocessor__')}
        model_instance = reconstruct_model_from_params(model_type, model_params_only)
        final_estimators.append((model_type, model_instance))


    final_full_pipeline = Pipeline([
        ('preprocessor', build_final_preprocessor(best_preprocessor_params, num_features, cat_features)),
        ('classifier', VotingClassifier(estimators=final_estimators, voting=best_voting_type, n_jobs=-1))
    ])

    final_full_pipeline.fit(X, y) # Fit on the entire original training data


    print(f"\n--- Final Model Configuration ({best_voting_type.upper()} Voting) ---")

    print("\nPreprocessor Parameters (from best single model):")
    for key, value in best_preprocessor_params.items():
        print(f"  {key}: {value}")

    print("\nBase Estimators and their Tuned Parameters (from Grid Search):")
    for model_type, results in top_models_info:
        print(f"  - {model_type.upper()}")
        model_params = {k: v for k, v in results['best_params'].items() if not k.startswith('preprocessor__')}
        for param_name, param_value in model_params.items():
            print(f"    {param_name}: {param_value}")

    # Save components from the final_full_pipeline
    fitted_preprocessor = final_full_pipeline.named_steps['preprocessor']
    trained_voting_model = final_full_pipeline.named_steps['classifier']

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

    # Find the saved voting model file
    model_files = [f for f in os.listdir(model_dir) if f.startswith('trained_') and f.endswith('_voting_model.joblib')]
    if not model_files:
        print(f"Error: No trained voting model file found in {model_dir}. Exiting.")
        sys.exit(1)
    model_path = os.path.join(model_dir, model_files[0]) # Assuming only one such file
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

    # Create a pipeline just for prediction using loaded components
    predict_pipeline = Pipeline([
        ('preprocessor', loaded_preprocessor),
        ('classifier', loaded_model)
    ])

    # Use the pipeline to make predictions
    predictions = predict_pipeline.predict(df_test)
    print(f"Generated {len(predictions)} predictions.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({'two_year_recid': predictions}).to_json(output_path, orient='records', lines=True)
    print(f"Predictions saved to: {output_path}")

    print("--- Prediction Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Train/Predict Recidivism Voting Ensemble Model with Grid Search")
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    parser_train = subparsers.add_parser('train', help='Tune models using Grid Search, create best voting ensemble, save preprocessor and model')
    parser_train.add_argument('--public_dir', type=str, default='./', help='Directory containing train_data/train.json')
    parser_train.add_argument('--model_dir', type=str, default ='./', help='Directory to save preprocessor.joblib and trained_voting_model.joblib')
    parser_train.add_argument('--n_cv_folds', type=int, default=5, help='Number of CV folds for Grid Search (default: 5)')

    parser_predict = subparsers.add_parser('predict', help='Make predictions using saved preprocessor and voting model')
    parser_predict.add_argument('--model_dir', type=str, default='./', help='Directory containing preprocessor.joblib and trained_..._voting_model.joblib')
    parser_predict.add_argument('--test_input_dir', type=str, default='./test_data', help='Directory containing test.json')
    parser_predict.add_argument('--output_path', type=str, default='./predict.json', help='File path for saving predictions')

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args.public_dir, args.model_dir, args.n_cv_folds)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.test_input_dir, args.output_path)

if __name__ == "__main__":
    main()