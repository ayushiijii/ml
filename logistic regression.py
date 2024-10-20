import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)  # number of training examples
    predictions = sigmoid(X @ theta)
    cost = - (1/m) * np.sum(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        predictions = sigmoid(X @ theta)
        errors = predictions - y
        gradient = (1/m) * (X.T @ errors)
        theta -= learning_rate * gradient
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history

def predict(X, theta, threshold=0.5):
    probabilities = sigmoid(X @ theta)
    return (probabilities >= threshold).astype(int)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def print_logistic_regression_equation(theta):
    b0 = theta[0]
    coefficients = theta[1:]
    terms = [f"{coeff:.4f} * x{i+1}" for i, coeff in enumerate(coefficients)]
    Z = f"{b0:.4f} + " + " + ".join(terms)
    print("Logistic Regression Equation:")
    print(f"p(y = 1 | X) = 1 / (1 + exp(-({Z})))")
    print(f"b0 (intercept): {b0:.4f}")
    for i, coeff in enumerate(coefficients, start=1):
        print(f"b{i} (coefficient for x{i}): {coeff:.4f}")

def make_prediction(theta, scaler, income, savings):
    # Prepare the input data
    input_data = np.array([[1, income, savings]])  # Add intercept term
    input_data_scaled = scaler.transform(input_data[:, 1:])  # Scale features
    input_data[:, 1:] = input_data_scaled
    prediction_prob = sigmoid(input_data @ theta)
    
    if prediction_prob > 0.5:
        print("Loan Sanctioned")
    else:
        print("Loan Not Sanctioned")

def load_data(file_path):
    df = pd.read_excel(file_path)  # Load the dataset
    # Check and strip column names of any leading/trailing spaces
    df.columns = df.columns.str.strip()
    
    # Ensure the required columns exist
    required_columns = ['Annual Income', 'Savings', 'target']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
    
    return df

def main():
    # Load the dataset
    file_path = r"C:\Users\Ayushi\Desktop\submissions\ML\dataset.xlsx"
    df = load_data(file_path)

    X = df[['Annual Income', 'Savings']].values
    y = df['target'].values

    # Add intercept term to feature matrix X
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale features
    scaler = StandardScaler()
    X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
    X_test[:, 1:] = scaler.transform(X_test[:, 1:])

    # Initialize parameters
    theta = np.zeros(X_train.shape[1])
    learning_rate = 0.01
    num_iterations = 1000

    # Perform gradient descent
    theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

    # Print logistic regression parameters
    print("Logistic Regression Parameters:")
    print_logistic_regression_equation(theta)

    # Make predictions
    train_predictions = predict(X_train, theta)
    test_predictions = predict(X_test, theta)
    train_accuracy = accuracy(y_train, train_predictions)
    test_accuracy = accuracy(y_test, test_predictions)

    # Print accuracy
    print("Training Accuracy:", train_accuracy, "%")
    print("Test Accuracy:", test_accuracy, "%")

    # Plot cost history
    plt.plot(range(num_iterations), cost_history, label='Cost')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.legend()
    plt.show()

    # Make predictions based on user input
    try:
        test_income = float(input("Enter Annual Income (Lakhs): "))
        test_savings = float(input("Enter Savings (Lakhs): "))
        make_prediction(theta, scaler, test_income, test_savings)
    except ValueError:
        print("Invalid input! Please enter numerical values.")

if __name__ == "__main__":
    main()
