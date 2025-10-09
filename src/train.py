import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from kaggle.api.kaggle_api_extended import KaggleApi

def load_dataset():
    """Download and load the Titanic dataset"""
    print("Downloading Titanic dataset from Kaggle...")
    
    # Initialize Kaggle API (no authentication required for public datasets)
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    dataset_path = '/app/data'
    os.makedirs(dataset_path, exist_ok=True)
    
    print("Downloading dataset files...")
    api.dataset_download_files('yasserh/titanic-dataset', path=dataset_path, unzip=True)
    print(f"Path to dataset files: {dataset_path}")
    
    # Find the CSV file in the downloaded path
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded dataset")
    
    # Load the first CSV file (usually Titanic.csv)
    data_path = os.path.join(dataset_path, csv_files[0])
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from: {data_path}")
    print(f"Dataset shape: {df.shape}")
    
    return df

def preprocess_data(df):
    """Preprocess the Titanic dataset"""
    # Select relevant columns
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    target = 'Survived'
    
    # Create a copy with selected columns
    df_clean = df[features + [target]].copy()
    
    print(f"\nOriginal dataset shape: {df_clean.shape}")
    print(f"Missing values:\n{df_clean.isnull().sum()}")
    
    # Handle missing values
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
    
    # Drop any remaining rows with missing values
    df_clean.dropna(inplace=True)
    
    print(f"\nDataset shape after cleaning: {df_clean.shape}")
    
    # Encode categorical variable (Sex)
    le = LabelEncoder()
    df_clean['Sex'] = le.fit_transform(df_clean['Sex'])
    
    # Save the label encoder for later use
    with open('/app/models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("Label encoder saved")
    
    return df_clean

def train_model(df):
    """Train a Random Forest classifier"""
    # Split features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(

        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

def save_model(model):
    """Save the trained model as a pickle file"""
    model_path = '/app/models/titanic_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("TITANIC SURVIVAL PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs('/app/models', exist_ok=True)
    
    # Load dataset
    df = load_dataset()
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Train model
    model = train_model(df_clean)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

