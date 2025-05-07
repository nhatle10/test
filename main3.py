import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
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


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from joblib import dump, load
from tqdm import tqdm

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


def run_train(public_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    # Load training data
    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    df = pd.read_json(train_path, lines=True)

    # Split features and label
    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    #X = engineer_features(X)

    # Identify categorical and numerical columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
        ]),  num_features),

            ('cat', Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ]), cat_features)
        ]
    )

    # Fit preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Extended param grid for DecisionTreeClassifier
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

    base_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_processed, y)

    best_model = grid_search.best_estimator_
    print(f"Best parameters:\n{grid_search.best_params_}\n")
    print("Training classification report:")
    print(classification_report(y, best_model.predict(X_processed)))

    # Save model and preprocessor
    dump(best_model, os.path.join(model_dir, 'trained_model.joblib'))
    dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))


def run_predict(model_dir, test_input_dir, output_path):
    # Load model and preprocessor
    model = load(os.path.join(model_dir, 'trained_model.joblib'))
    preprocessor = load(os.path.join(model_dir, 'preprocessor.joblib'))

    # Load test data
    test_path = os.path.join(test_input_dir, 'test.json')
    df_test = pd.read_json(test_path, lines=True)

    #df_test = engineer_features(df_test)

    # Transform and predict
    X_test = preprocessor.transform(df_test)
    preds = model.predict(X_test)

    # Save predictions
    pd.DataFrame({'two_year_recid': preds}).to_json(output_path, orient='records', lines=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Train command
    parser_train = subparsers.add_parser('train', help='Tune models using Grid Search, create best voting ensemble, save preprocessor and model')
    parser_train.add_argument('--public_dir', type=str, default='./', help='Directory containing train_data/train.json')
    parser_train.add_argument('--model_dir', type=str, default ='./', help='Directory to save preprocessor.joblib and trained_voting_model.joblib')

    # Predict command
    parser_predict = subparsers.add_parser('predict', help='Make predictions using saved preprocessor and voting model')
    parser_predict.add_argument('--model_dir', type=str, default='./', help='Directory containing preprocessor.joblib and trained_..._voting_model.joblib')
    parser_predict.add_argument('--test_input_dir', type=str, default='./test_data', help='Directory containing test.json')
    parser_predict.add_argument('--output_path', type=str, default='./predict.json', help='File path for saving predictions')

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args.public_dir, args.model_dir)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.test_input_dir, args.output_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
