# =============================================================
# 🏠 HOUSE PRICE PREDICTOR
# =============================================================
# WHAT THIS PROJECT DOES:
# Takes house features (rooms, area, location etc.)
# and predicts the price of the house.
# We use California Housing Dataset (built into sklearn)
# so no need to download anything!
# =============================================================


# ===== IMPORTS =====
import pandas as pd               # for working with data in table format
import numpy as np                # for numerical operations
import matplotlib.pyplot as plt   # for plotting graphs
import seaborn as sns             # for beautiful visualizations

from sklearn.datasets import fetch_california_housing     # our dataset
from sklearn.model_selection import train_test_split      # split data into train/test
from sklearn.linear_model import LinearRegression         # simple regression model
from sklearn.ensemble import RandomForestRegressor        # powerful tree-based model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler          # normalize/scale features
import warnings
warnings.filterwarnings('ignore')  # hide unnecessary warnings

print("✅ All libraries imported!")


# =============================================================
# STEP 1 — LOAD DATASET
# =============================================================
# fetch_california_housing() gives us a dataset with:
# - 20,640 houses in California
# - 8 features per house (rooms, population, location etc.)
# - Target: median house price (in $100,000s)
# =============================================================

housing = fetch_california_housing()

# Convert to pandas DataFrame (table format — easier to work with)
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target  # add the target column (price)

print("\n📊 Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())


# =============================================================
# STEP 2 — EXPLORE THE DATA (EDA)
# =============================================================
# EDA = Exploratory Data Analysis
# Before training any model, always understand your data first!
# Check: any missing values? what are the ranges? distributions?
# =============================================================

print("\n📈 Basic Statistics:")
print(df.describe())  # shows min, max, mean, std for each column

print("\n🔍 Missing Values:")
print(df.isnull().sum())  # check if any column has missing data


# =============================================================
# STEP 3 — VISUALIZE THE DATA
# =============================================================
# Visualization helps us understand:
# 1. Distribution of house prices
# 2. Which features are most correlated with price
# =============================================================

# --- Plot 1: Distribution of House Prices ---
plt.figure(figsize=(8, 4))
sns.histplot(df['Price'], bins=50, kde=True, color='steelblue')
# histplot = histogram (bar chart showing how prices are distributed)
# kde=True = adds a smooth curve on top
plt.title('Distribution of House Prices')
plt.xlabel('Price (in $100,000s)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('price_distribution.png')  # save the plot as image
plt.show()
print("\n✅ Plot 1 saved: price_distribution.png")

# --- Plot 2: Correlation Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
# corr() = calculates how much each feature relates to others
# annot=True = show numbers inside the boxes
# cmap='coolwarm' = red for high correlation, blue for low
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("✅ Plot 2 saved: correlation_heatmap.png")

# --- Plot 3: Top Features vs Price ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# MedInc = Median Income of the area (most important feature!)
axes[0].scatter(df['MedInc'], df['Price'], alpha=0.3, color='steelblue')
axes[0].set_xlabel('Median Income')
axes[0].set_ylabel('Price')
axes[0].set_title('Income vs Price')

# AveRooms = Average number of rooms
axes[1].scatter(df['AveRooms'], df['Price'], alpha=0.3, color='coral')
axes[1].set_xlabel('Average Rooms')
axes[1].set_ylabel('Price')
axes[1].set_title('Rooms vs Price')
axes[1].set_xlim(0, 20)  # limit x axis to remove outliers

plt.tight_layout()
plt.savefig('feature_vs_price.png')
plt.show()
print("✅ Plot 3 saved: feature_vs_price.png")


# =============================================================
# STEP 4 — PREPARE DATA FOR TRAINING
# =============================================================
# X = input features (what we give the model)
# y = target (what we want the model to predict)
# Split: 80% training, 20% testing
# Scale: make all features on same scale (important for LinearRegression)
# =============================================================

X = df.drop('Price', axis=1)  # everything except price
y = df['Price']                # only price

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# StandardScaler: transforms features so they all have mean=0, std=1
# Why? Linear Regression is sensitive to feature scale
# Example: 'Income' ranges 0-15, 'Population' ranges 0-35000
# Without scaling, large numbers dominate the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train, transform train
X_test_scaled = scaler.transform(X_test)        # only transform test (never fit on test!)

print("✅ Data scaled and ready!")


# =============================================================
# STEP 5 — TRAIN MODEL 1: LINEAR REGRESSION
# =============================================================
# Linear Regression: finds the best straight line
# that fits the relationship between features and price
# Simple, fast, interpretable — good baseline model
# =============================================================

print("\n📐 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)

# Metrics:
# MAE  = Mean Absolute Error (average error in same unit as price)
# RMSE = Root Mean Squared Error (penalizes large errors more)
# R²   = How much variance the model explains (1.0 = perfect, 0 = bad)
lr_mae  = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2   = r2_score(y_test, lr_predictions)

print(f"Linear Regression Results:")
print(f"  MAE  : {lr_mae:.4f} (${lr_mae*100000:.0f} average error)")
print(f"  RMSE : {lr_rmse:.4f}")
print(f"  R²   : {lr_r2:.4f} ({lr_r2*100:.1f}% variance explained)")


# =============================================================
# STEP 6 — TRAIN MODEL 2: RANDOM FOREST
# =============================================================
# Random Forest: builds many decision trees and averages them
# More powerful than Linear Regression
# Can capture complex non-linear patterns
# =============================================================

print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# n_estimators=100 means we build 100 decision trees
rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaling

rf_predictions = rf_model.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2   = r2_score(y_test, rf_predictions)

print(f"Random Forest Results:")
print(f"  MAE  : {rf_mae:.4f} (${rf_mae*100000:.0f} average error)")
print(f"  RMSE : {rf_rmse:.4f}")
print(f"  R²   : {rf_r2:.4f} ({rf_r2*100:.1f}% variance explained)")


# =============================================================
# STEP 7 — COMPARE MODELS
# =============================================================

print("\n📊 Model Comparison:")
print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
print("-" * 55)
print(f"{'Linear Regression':<25} {lr_mae:<10.4f} {lr_rmse:<10.4f} {lr_r2:<10.4f}")
print(f"{'Random Forest':<25} {rf_mae:<10.4f} {rf_rmse:<10.4f} {rf_r2:<10.4f}")


# =============================================================
# STEP 8 — FEATURE IMPORTANCE
# =============================================================
# Which features matter most for predicting price?
# Random Forest gives us this for free!
# =============================================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔑 Feature Importance (Random Forest):")
print(feature_importance)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance - Which features affect price most?')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("✅ Plot 4 saved: feature_importance.png")


# =============================================================
# STEP 9 — PREDICT ON A CUSTOM HOUSE
# =============================================================
# Let's predict the price of a sample house!
# Feature order: MedInc, HouseAge, AveRooms, AveBedrms,
#                Population, AveOccup, Latitude, Longitude
# =============================================================

sample_house = np.array([[5.0, 20.0, 6.0, 1.0, 1200.0, 3.0, 37.0, -122.0]])
predicted_price = rf_model.predict(sample_house)[0]

print(f"\n🏠 Sample House Prediction:")
print(f"   Features: Income=5.0, Age=20yrs, Rooms=6, Population=1200")
print(f"   Predicted Price: ${predicted_price * 100000:,.0f}")

print("\n🎉 Project Complete!")