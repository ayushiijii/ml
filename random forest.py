# Import the necessary libraries
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Load the iris dataset
flower_dataset = datasets.load_iris()

# Print basic information about the dataset
print(flower_dataset.target_names)  # Prints the target class names (Setosa, Versicolor, Virginica)
print(flower_dataset.feature_names)  # Prints the feature names (sepal/petal lengths and widths)
print(flower_dataset.data[0:5])  # Prints the first 5 rows of feature data
print(flower_dataset.target)  # Prints the target labels (species)

# Create a DataFrame from the iris data
flower_df = pd.DataFrame({
    'sepal_length': flower_dataset.data[:, 0],
    'sepal_width': flower_dataset.data[:, 1],
    'petal_length': flower_dataset.data[:, 2],
    'petal_width': flower_dataset.data[:, 3],
    'species_label': flower_dataset.target
})

# Display the first few rows of the DataFrame
print(flower_df.head())

# Split the data into features and labels
input_features = flower_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Features
output_labels = flower_df['species_label']  # Labels (species)

# Split the dataset into training and testing sets (70% training, 30% testing)
input_train, input_test, output_train, output_test = train_test_split(
    input_features, output_labels, test_size=0.3, random_state=42
)

# Initialize the RandomForestClassifier with 100 trees
forest_classifier = RandomForestClassifier(n_estimators=100)

# Fit the classifier on the training data
forest_classifier.fit(input_train, output_train)

# Make predictions on the test data
predicted_labels = forest_classifier.predict(input_test)

# Calculate and print the accuracy of the model
print("Classification Accuracy: ", metrics.accuracy_score(output_test, predicted_labels))

# Make a prediction on new data (sepal/petal measurements)
new_sample = pd.DataFrame([[5, 2.5, 3.5, 1]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
species_prediction = forest_classifier.predict(new_sample)

# Output the predicted species based on the model's prediction
if species_prediction[0] == 0:
    print('Predicted Species: Setosa')
elif species_prediction[0] == 1:
    print('Predicted Species: Versicolor')
else:
    print('Predicted Species: Virginica')
