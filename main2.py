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
        self.transformers = {}

    def fit(self, X, y=None):
        for group in X[self.label].unique():
            self.transformers[group] = PowerTransformer(method='yeo-johnson', standardize=True)
            self.transformers[group].fit(
                X.loc[X[self.label] == group, self.col].values.reshape(-1, 1)
            )
        return self

    def transform(self, X, y=None):
        C = X.copy()
        for group in X[self.label].unique():
            C.loc[X[self.label] == group, self.col] = self.transformers[group].transform(
                X.loc[X[self.label] == group, self.col].values.reshape(-1, 1)
            )
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

    df.loc[df['race'].isin(['Native American', 'Asian']), 'race'] = 'Other'

    # Điền giá trị NaN bằng 0 cho các cột vi phạm vị thành niên
    df[['juv_fel_count', 'juv_misd_count', 'juv_other_count']] = df[[
        'juv_fel_count', 'juv_misd_count', 'juv_other_count'
    ]].fillna(0)

    df['priors_count'] = df['priors_count'].fillna(0)

    df['juv_count'] = df[["juv_fel_count", "juv_misd_count", "juv_other_count"]].sum(axis=1)

    df = df.drop(["juv_fel_count", "juv_misd_count", "juv_other_count"], axis=1)

    # Age category
    # df['age_cat'] = df['age'].apply(categorize_age)
    # df.drop(columns=['age'], inplace=True)

    # Tổng số vi phạm vị thành niên
    # df['juv_total_count'] = df['juv_fel_count'] + df['juv_misd_count'] + df['juv_other_count']

    # # Có từng phạm pháp khi vị thành niên không?
    # df['is_juvenile_offender'] = (df['juv_total_count'] > 0).astype(int)

    # # Đã từng phạm pháp trước đó?
    # df['is_repeat_offender'] = (df['priors_count'] > 0).astype(int)

    # # Có phạm tội nghiêm trọng không?
    # df['is_felony'] = (df['c_charge_degree'] == 'F').astype(int)

    # df.drop(columns=['c_charge_degree'], inplace=True)

    #df.drop(columns=['juv_fel_count', 'juv_misd_count', 'juv_other_count'], inplace=True)

    return df


def run_train(public_dir, model_dir, model_name='decision_tree'):
    os.makedirs(model_dir, exist_ok=True)

    # Load training data
    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    df = pd.read_json(train_path, lines=True)

    #df = engineer_features(df)

    # Split features and label
    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    # Identify categorical and numerical columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    if 'priors_count' in num_features:
        num_features.remove('priors_count')

    # Add NormalizeColumnByLabel as a step in the full pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), num_features),

            ('cat', Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ]), cat_features)
        ]
    )

    # Fit preprocessing
    X_processed = preprocessor.fit_transform(X)

    # MODEL SELECTION
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
        # param_grid = {
        #     'criterion': ['gini', 'entropy', 'log_loss'],
        #     'splitter': ['best', 'random'],
        #     'max_depth': [None, 5, 10, 20, 30],
        #     'min_samples_split': [2, 5, 10, 20],
        #     'min_samples_leaf': [1, 2, 4, 6],
        #     'max_features': [None, 'sqrt', 'log2'],
        #     'max_leaf_nodes': [None, 10, 20, 50],
        #     'min_weight_fraction_leaf': [0.0, 0.01, 0.05],
        #     'class_weight': [None, 'balanced'],
        #     'ccp_alpha': [0.0, 0.001, 0.01, 0.1]  
        # }
    elif model_name == 'catboost':
        base_model = CatBoostClassifier(verbose=0, random_state=42)
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
    # elif model_name == 'random_forest':
    #     base_model = RandomForestClassifier(random_state=42)
    #     param_grid = {
    #         'n_estimators': [100, 200, 300],
    #         'max_depth': [None, 10, 20, 30],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 2, 4],
    #         'max_features': ['sqrt', 'log2', None],
    #         'bootstrap': [True, False],
    #         'class_weight': [None, 'balanced']
    #     }
    elif model_name == 'random_forest':
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced'],
            'criterion': ['gini', 'entropy', 'log_loss'],  # thêm tiêu chí phân tách
            'max_leaf_nodes': [None, 20, 50, 100],         # giới hạn số lá
            'min_weight_fraction_leaf': [0.0, 0.01, 0.02], # ngưỡng tối thiểu cho mỗi lá
            'ccp_alpha': [0.0, 0.001, 0.01]                # pruning
        }
    elif model_name == 'adaboost':
        base_tree = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
        base_model = AdaBoostClassifier(base_estimator=base_tree, random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'base_estimator__max_depth': [1, 2, 3],
            'base_estimator__class_weight': [None, 'balanced']
        }
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
            'booster': ['gbtree', 'gblinear']  # Chọn loại booster
        }
    elif model_name == 'logistic_regression':
        base_model = LogisticRegression(solver='liblinear', random_state=42)
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'fit_intercept': [True, False],
            'max_iter': [100, 200, 500],
            'solver': ['liblinear']  # Vì 'liblinear' hỗ trợ cả 'l1' và 'l2'
        }
    elif model_name == 'lgbm':
        base_model = LGBMClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [-1, 10, 20, 30],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'class_weight': [None, 'balanced']
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Grid search
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
    parser_train.add_argument('--model_name', type=str, default='decision_tree', choices=['decision_tree', 'catboost', 'random_forest', 'adaboost', 'xgboost', 'logistic_regression', 'lgbm'], help='Model to train')

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
