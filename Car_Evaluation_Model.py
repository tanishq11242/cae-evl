import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_car_evaluation_data(csv_file_path: str) -> pd.DataFrame:
    """
    Reads the Car Evaluation dataset CSV (without header) into a DataFrame
    and assigns descriptive column names.
    """
    column_names = ['buying_price', 'maintenance_cost', 'num_doors',
                    'passenger_capacity', 'luggage_boot_size', 'safety_rating',
                    'acceptability']
    car_df = pd.read_csv(csv_file_path, header=None, names=column_names)
    return car_df


def explore_data(car_df: pd.DataFrame) -> None:
    """
    Displays sample rows and class distribution, and saves a bar chart.
    """
    print("=== Data Preview ===")
    print(car_df.head(), "\n")

    acceptability_counts = car_df['acceptability'].value_counts()
    print("=== Acceptability Distribution ===")
    print(acceptability_counts, "\n")

    # Plot and save distribution
    fig, ax = plt.subplots()
    acceptability_counts.plot(kind='bar', ax=ax)
    ax.set_title("Car Acceptability Class Distribution")
    ax.set_xlabel("Acceptability")
    ax.set_ylabel("Number of Instances")
    plt.tight_layout()
    plt.savefig('acceptability_distribution.png')
    plt.close()


def split_and_preprocess(car_df: pd.DataFrame):
    """
    Separates features and target, encodes the target labels, splits into train/test,
    and constructs a preprocessing pipeline for categorical features.
    Returns:
      X_train, X_test, y_train, y_test, preprocessor, label_encoder
    """
    # Separate features and target
    feature_df = car_df.drop('acceptability', axis=1)
    target_series = car_df['acceptability']

    # Encode categorical target to numeric labels
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target_series)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        encoded_target,
        test_size=0.2,
        random_state=42,
        stratify=encoded_target
    )

    # One-hot encode all categorical features
    categorical_columns = feature_df.columns.tolist()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot_encoding', one_hot_encoder, categorical_columns)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor, label_encoder


def train_and_select_best_model(X_train, X_test, y_train, y_test, preprocessor):
    """
    Trains multiple models, evaluates their accuracy on the test set,
    and returns the name and pipeline of the highest-performing model.
    """
    # Define candidate models
    model_constructors = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Linear Regression': LinearRegression()
    }

    model_pipelines = {}
    model_accuracies = {}

    for model_name, model_instance in model_constructors.items():
        # Create a pipeline: preprocessing + model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('estimator', model_instance)
        ])
        # Fit to training data
        pipeline.fit(X_train, y_train)
        # Predict on test data
        raw_predictions = pipeline.predict(X_test)

        # If regression model, round and clip predictions to valid label indices
        if model_name == 'Linear Regression':
            predictions = np.rint(raw_predictions).astype(int)
            predictions = np.clip(predictions, y_train.min(), y_train.max())
        else:
            predictions = raw_predictions

        # Compute accuracy
        accuracy = accuracy_score(y_test, predictions)
        model_pipelines[model_name] = pipeline
        model_accuracies[model_name] = accuracy
        print(f"{model_name} Accuracy: {accuracy:.4f}")

    # Select best model by accuracy
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_model_pipeline = model_pipelines[best_model_name]
    best_accuracy = model_accuracies[best_model_name]
    print(f"\nBest Model Selected: {best_model_name} (Accuracy: {best_accuracy:.4f})")

    return best_model_name, best_model_pipeline


def save_trained_model(model_pipeline: Pipeline, label_encoder: LabelEncoder,
                       output_filename: str = 'car_acceptability_model.joblib') -> None:
    """
    Persists the trained model pipeline and label encoder to a Joblib file.
    """
    joblib.dump({'pipeline': model_pipeline, 'label_encoder': label_encoder}, output_filename)
    print(f"Trained model and label encoder saved to '{output_filename}'")


def main():
    # Ensure correct usage
    if len(sys.argv) != 2:
        print("Usage: python car_evaluation_project.py /path/to/car_data.csv")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    # Load and explore the data
    car_dataframe = load_car_evaluation_data(csv_file_path)
    explore_data(car_dataframe)

    # Preprocess and split
    X_train, X_test, y_train, y_test, data_preprocessor, label_encoder = \
        split_and_preprocess(car_dataframe)

    # Train models and select the best
    best_model_name, best_pipeline = train_and_select_best_model(
        X_train, X_test, y_train, y_test, data_preprocessor
    )

    # Save the selected model
    save_trained_model(best_pipeline, label_encoder)


if __name__ == '__main__':
    main()