import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data.csv")

# Display first 5 rows
print(data.head())

# Split dataset
from sklearn.model_selection import train_test_split

X = data[['Hours']]
y = data['Scores']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("\nPredicted Values:")
print(predictions)

# Accuracy
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", error)

# Predict all values for graph
all_predictions = model.predict(X)

# Graph
plt.scatter(data['Hours'], data['Scores'], color='blue')

plt.plot(data['Hours'], all_predictions, color='red')

plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.title("Linear Regression Prediction")

plt.show()