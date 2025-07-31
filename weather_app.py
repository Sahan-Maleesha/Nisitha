import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:/Users/Sahan Maleesha/Downloads/Train Data.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDescriptive statistics for numerical columns:")
print(df.describe())

print("\nDescriptive statistics for categorical columns:")
print(df.describe(include=['object']))

if 'RainTomorrow' in df.columns:
    print("\nValue counts for RainTomorrow:")
    print(df['RainTomorrow'].value_counts())

missing_counts = df.isnull().sum()
print("Missing values per column:")
print(missing_counts)

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols].hist(figsize=(15, 10), bins=20, color='skyblue')
plt.suptitle('Histograms of Numerical Features')
# plt.show()

if 'RainTomorrow' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='RainTomorrow', data=df)
    plt.title('Target Variable Distribution (RainTomorrow)')
    # plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap (Numerical Features)')
# plt.show()

# 5. Boxplots for Outliers (Numerical Features)
for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    # plt.show()

# 6. Bar Plots for Categorical Features
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f'Countplot of {col}')
    # plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Gradient Boosting Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

