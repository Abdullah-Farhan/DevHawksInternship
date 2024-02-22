# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the data from CSV file
data = pd.read_csv('dataset.csv')

# Replace 'male' with 0 and 'female' with 1 in the 'Sex' column
data['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})

# Drop rows with missing values
data.dropna(inplace=True)

# Filling missing fields
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']])

# Creating feature matrix X and target vector y
X = X_imputed
y = data['Survived']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Creating and fitting the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
