import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Load dataset
# ---------------------------
df_before = pd.read_csv("../data/titanic.csv")

# Rename columns for clarity
rename_map = {
    'sibsp': 'siblings_spouses_aboard',
    'parch': 'parents_children_aboard',
    'fare': 'ticket_fare',
    'pclass': 'passenger_class',
    'embarked': 'port_of_embarkation'
}
df_before.rename(columns=rename_map, inplace=True)

# Keep a copy before cleaning
df = df_before.copy()

# ---------------------------
# BEFORE CLEANING PLOTS
# ---------------------------
num_cols_before = df_before.select_dtypes(include=[np.number]).columns
for col in num_cols_before:
    plt.figure(figsize=(6,3))
    sns.histplot(df_before[col], kde=True, bins=30, color='#8e7cc3')  # purple
    plt.title(f"[Before] Distribution of {col}", color='#8e7cc3')
    plt.show()

# ---------------------------
# DATA CLEANING
# ---------------------------
# Fill missing Age with median
df['age'] = df['age'].fillna(df['age'].median())

# Fill missing Embarked with mode
df['port_of_embarkation'] = df['port_of_embarkation'].fillna(df['port_of_embarkation'].mode()[0])

# Drop Cabin (too many missing values)
if 'cabin' in df.columns:
    df.drop(columns=['cabin'], inplace=True)

# Drop rows with missing Embarked (after filling)
df.dropna(subset=['port_of_embarkation'], inplace=True)

# Rename again (in case new df needs it)
df.rename(columns=rename_map, inplace=True)

# ---------------------------
# AFTER CLEANING PLOTS
# ---------------------------
num_cols_after = df.select_dtypes(include=[np.number]).columns
for col in num_cols_after:
    plt.figure(figsize=(6,3))
    sns.histplot(df[col], kde=True, bins=30, color='orange')
    plt.title(f"[After] Distribution of {col}", color='orange')
    plt.show()

# ---------------------------
# Replace Survived with text labels
# ---------------------------
df['survived'] = df['survived'].map({0: 'Not Survived', 1: 'Survived'})
df_before['survived'] = df_before['survived'].map({0: 'Not Survived', 1: 'Survived'})

# ---------------------------
# EXPLORATORY ANALYSIS
# ---------------------------
# Survival count
plt.figure(figsize=(5,3))
sns.countplot(x='survived', data=df, palette='Set2')
plt.title("Survival Count")
plt.show()

# Survival by gender
plt.figure(figsize=(5,3))
sns.countplot(x='sex', hue='survived', data=df, palette='Set2')
plt.title("Survival by Gender")
plt.show()

# Survival by passenger class
plt.figure(figsize=(5,3))
sns.countplot(x='passenger_class', hue='survived', data=df, palette='Set2')
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution by survival
plt.figure(figsize=(6,4))
sns.kdeplot(df[df['survived'] == 'Survived']['age'], shade=True, label='Survived')
sns.kdeplot(df[df['survived'] == 'Not Survived']['age'], shade=True, label='Not Survived')
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.legend()
plt.show()