# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Exploratory Data Analysis (EDA)
def eda():
    print("Dataset Overview:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe())

    # Visualizing the distribution of 'Outcome'
    sns.countplot(x='Outcome', data=df)
    plt.title('Outcome Distribution')
    plt.show()

    # Visualize relationships between variables
    sns.pairplot(df)
    plt.show()

# Preprocessing
def preprocess_data():
    # Split into features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Model Training and Evaluation
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Main function to run the process
if __name__ == "__main__":
    eda()
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data()
    train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
