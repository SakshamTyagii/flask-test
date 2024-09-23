import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the CSV data
data = pd.read_csv(r'C:\Users\saksham tyagi\OneDrive\Desktop\agri+\data.csv')

# Split the data into features (X) and label (y)
y = data['label']
X = data.drop(['label'], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model and feature names
joblib.dump((model, X.columns), 'model.pkl')

