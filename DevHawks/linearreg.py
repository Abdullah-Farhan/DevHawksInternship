# Importing necessary library
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv')

# Converting string rows to numeric data
data['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})

# Filling missing fields by taking mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']])

# Model Training
model = LinearRegression()
model.fit(X_imputed, data['Survived'])

# Prediction
y_pred = model.predict(X_imputed)

print(y_pred)

# Output
plt.scatter(data['Survived'], y_pred)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True Values vs Predicted Values")
plt.show()