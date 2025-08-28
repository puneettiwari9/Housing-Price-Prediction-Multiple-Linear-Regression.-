import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import sqlite3  # for database

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv(r"D:\data analyst\MACHINE LEARNING\ML PROGRAMS USING PYTHON\LINEAR REGRESSION IN ML\project in ml linear reg\Kaggle_housingdata.csv")

# Using describe to get mean and std for numeric columns
desc = data.describe()
print("Mean values:\n", desc.loc['mean'])
print("\nStandard deviations:\n", desc.loc['std'])

# Convert categorical 'mainroad' to numeric
data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})

# Select features and target
features = ['area', 'bedrooms', 'bathrooms', 'parking', 'mainroad']
X = data[features]
y = data['price']

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# -----------------------------
# 3D Plot for area, bedrooms, bathrooms vs predicted price
# -----------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
x = X_test['area']
y_ = X_test['bedrooms']
z = X_test['bathrooms']
c = y_pred  # Predicted price (color and height)

scatter = ax.scatter(x, y_, z, c=c, cmap='viridis', s=50, alpha=0.8)
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Predicted Price')

ax.set_xlabel('Area')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Bathrooms')
ax.set_title('3D scatter plot: Area, Bedrooms, Bathrooms vs Predicted Price')

plt.show()

# -----------------------------
# User input function
# -----------------------------
def yes_no_to_int(value):
    return 1 if value.lower() == 'yes' else 0

# User enters input
area = int(input("Enter area in feets (numeric): "))
bedrooms = int(input("Enter number of bedrooms (numeric): "))
bathrooms = int(input("Enter number of bathrooms (numeric): "))
parking = int(input("Enter number of parking spaces (numeric): "))
mainroad_input = input("Is there a main road? (yes/no): ")
mainroad = yes_no_to_int(mainroad_input)

# Create DataFrame for prediction
sample = pd.DataFrame([[area, bedrooms, bathrooms, parking, mainroad]], 
                      columns=['area', 'bedrooms', 'bathrooms', 'parking', 'mainroad'])

# Predict price
sample_pred = model.predict(sample)[0]

print(f"Predicted Price for input {sample.values.tolist()[0]}: {sample_pred:.2f}")
