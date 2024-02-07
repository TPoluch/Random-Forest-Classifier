import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Simulating some data
np.random.seed(42)  # For reproducibility

# Features
job_satisfaction = np.random.randint(1, 5, 1000)  # Ratings from 1 to 5
length_of_assignments = np.random.randint(1, 10, 1000)  # In months
specialties = np.random.randint(1, 20, 1000)  # Assuming 20 different specialties
performance_reviews = np.random.randint(1, 5, 1000)  # Ratings from 1 to 5

# Target variable: 1 for stayed, 0 for left
stayed_beyond_initial = (job_satisfaction + length_of_assignments / 2 + performance_reviews >= 7).astype(int)

# Creating a DataFrame
data = pd.DataFrame({
    'Job Satisfaction': job_satisfaction,
    'Length of Assignments': length_of_assignments,
    'Specialties': specialties,
    'Performance Reviews': performance_reviews,
    'Stayed Beyond Initial': stayed_beyond_initial
})

# Splitting the dataset into training and testing sets
X = data.drop('Stayed Beyond Initial', axis=1)
y = data['Stayed Beyond Initial']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predicting on the test set
y_pred = clf.predict(X_test)

# Open the CSV file in write mode
csvfile = open('data.csv', 'w')
writer = csv.writer(csvfile)
writer.writerow(
    ['Job Satisfaction', 'Length of Assignments', 'Specialties', 'Performance Reviews', 'Stayed Beyond Initial'])
for row in data.values:
    writer.writerow(row)

# Evaluating the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
