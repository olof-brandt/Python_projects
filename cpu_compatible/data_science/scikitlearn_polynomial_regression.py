"""
This script demonstrates polynomial regression using scikit-learn in Python.
It fits a quadratic (degree 2) polynomial to a set of data points and visualizes
both the original data and the fitted polynomial curve.

The process involves:
- Creating sample data points
- Transforming input features to include polynomial terms
- Fitting a linear regression model to the polynomial features
- Making predictions on the input data
- Plotting the original data points and the regression curve
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate sample data points
x = np.arange(0, 30)
y = [3, 4, 5, 7, 10, 8, 9, 10, 10, 23, 27, 44, 50, 63, 67, 60, 62, 70, 75, 88, 81, 87, 95, 100, 108, 135, 151, 160, 169, 179]

# Plot the original data points
plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.title("Original Data Points")
# Uncomment the next line if you want to see the scatter plot separately
# plt.show()

# Transform the input data to include polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
# Reshape x to a 2D array for sklearn, then transform
poly_features = poly.fit_transform(x.reshape(-1, 1))

# Initialize and fit the linear regression model to the polynomial features
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)

# Predict y values using the trained model
y_predicted = poly_reg_model.predict(poly_features)

# Print the predicted y values
print("Predicted y-values:", y_predicted)

# Plot the original data and the polynomial regression curve
plt.figure(figsize=(10, 6))
plt.title("Polynomial Regression (Degree 2)", size=16)
plt.scatter(x, y, label='Original Data')
plt.plot(x, y_predicted, color='red', label='Regression Fit')
plt.legend()
plt.show()
