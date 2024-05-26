import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
def load_data():
    df = pd.read_csv("Titanic.csv")
    return df

# Data preprocessing
def preprocess_data(df):
    # Drop irrelevant columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Convert categorical variables to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    return df

# Split data into features and target
def split_data(df):
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return X, y

# Train model with hyperparameter tuning using Grid Search
def train_model_with_grid_search(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
    
    # Define hyperparameters for grid search
    param_grid = {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__max_depth': [3, 4, 5, 6],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__subsample': [0.5, 0.8, 1.0]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    return grid_search

# Train model using Random Forest
def train_random_forest(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline

# Train model using AdaBoost
def train_adaboost(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', AdaBoostClassifier(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline

# Perform cross-validation
def perform_cross_validation(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])

    # Perform cross-validation with larger number of folds
    scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')

    return scores

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Main function
def main():
    # Load data
    df = load_data()

    # Preprocess data
    df = preprocess_data(df)

    # Split data
    X, y = split_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with hyperparameter tuning using Grid Search
    grid_search_model = train_model_with_grid_search(X_train, y_train)

    # Train Random Forest model
    random_forest_model = train_random_forest(X_train, y_train)

    # Train AdaBoost model
    adaboost_model = train_adaboost(X_train, y_train)

    # Perform cross-validation
    cross_val_scores = perform_cross_validation(X, y)

    # Evaluate models
    print("Gradient Boosting Classifier:")
    evaluate_model(grid_search_model, X_test, y_test)

    print("\nRandom Forest Classifier:")
    evaluate_model(random_forest_model, X_test, y_test)

    print("\nAdaBoost Classifier:")
    evaluate_model(adaboost_model, X_test, y_test)

    print("\nCross-Validation Scores:", cross_val_scores)
    print("Mean Cross-Validation Score:", np.mean(cross_val_scores))

if __name__ == "__main__":
    main()
