import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("iris.csv")

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Encode the 'species' column
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Display the full preprocessed data
print("\nPreprocessed Data (All Rows):")
print(df.to_string(index=False))

# Save cleaned data to a new CSV file
df.to_csv("iris_cleaned.csv", index=False)
print("\n✅ Cleaned data saved as 'iris_cleaned.csv'.")

# Split the data
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
