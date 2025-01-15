import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import xgboost as xgb
from xgboost import cv, DMatrix

from feature_eng import extract_features
from data_processing import process_data

# TODO:

#! overfitting in baseline models - fix using data shuffling, augmentation, dropout/regularisation/batch normalisation

def load_dataset(path):

    print("Loading dataset...")
    
    features_path = r'C:\Users\adity\Projects\CRISPR-ML\data\processed\features_guideseq.csv'

    extract_features(path, features_path);

    data = pd.read_csv(features_path)
    X = data.drop('Active', axis = 1)
    y = data['Active'].values.ravel()
    print(y.shape)
    return X, y

def one_hot(sequence):
    mapping = {'A' : [1,0,0,0], 'C' : [0,1,0,0], 'G' : [0,0,1,0], 'T' : [0,0,0,1]}
    encoded = [mapping[char] for char in sequence if char in mapping]
    return np.array(encoded).flatten()

def one_hot_encode(X, sequence_cols):
    encoded_features = []

    for col in sequence_cols:
        encoded = X[col].apply(one_hot) 
        max_length = max(encoded.map(len))

        encoded = encoded.apply(lambda sequence: np.pad(sequence, (0, max_length - len(sequence)), constant_values = 0))
        encoded_features.append(np.stack(encoded))

    encoded_features = np.concatenate(encoded_features, axis = 1)
    X = X.drop(columns = sequence_cols).reset_index(drop = True)
    X = pd.concat([X, pd.DataFrame(encoded_features)], axis=1)
    X.columns = X.columns.astype(str)

    return X


def split_dataset(X, y, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train, X_test, y_test):

    print("Training Logistic Regression model (baseline model 1) w/ Cross-Validation...")
    model = LogisticRegression(max_iter = 1000, random_state = 42, C = 0.1, penalty = 12)
    scores = cross_val_score(model, X_train, y_train, cv = 5, scoring = 'accuracy')

    print("Cross-validated accuracy: ", scores.mean())
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("ROC-AUC Score:", roc_auc_score(y_test, probabilities))
    print("Classification Report:", classification_report(y_test, predictions))

    return model

def train_xgboost(X_train, y_train, X_test, y_test):

    print("Training XGBoost model (baseline model 2)...")

    d_train = xgb.DMatrix(X_train, label = y_train)
    d_test = xgb.DMatrix(X_test, label = y_test)

    params = {
        'objective' : 'binary:logistic',
        'eval_matric' : 'auc',
        'max_depth' : 6,
        'learning_rate' : 0.1,
        'lambda' : 1.0,
        'alpha' : 1.0,
        'seed' : 42
    }

    evals = [(d_train, 'train'), (d_test, 'test')]

    model = xgb.train(
        params,
        d_train,
        num_boost_round = 100,
        evals = evals,
        early_stopping_rounds = 10
    )

    predictions = model.predict(d_test) > 0.5
    probabilities = model.predict(d_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("ROC-AUC Score:", roc_auc_score(y_test, probabilities))
    print("Classification Report:", classification_report(y_test, predictions))

    return model

def main():
    path = r'C:\Users\adity\Projects\CRISPR-ML\data\raw\guideseq.csv'
    X, y = load_dataset(path)

    X = one_hot_encode(X, ['On', 'Off'])

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    #LR_model = train_logistic_regression(X_train, y_train, X_test, y_test)
    XGB_model = train_xgboost(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()

