import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# LOAD DATA
# =====================
file_path = r"D:\SEMESTER 6\Mining\P\diabetes.csv"
df = pd.read_csv(file_path)

# =====================
# DATA CLEANING
# =====================
X = df.drop("Outcome", axis=1) #data input (glukosa, BMI, dll)
y = df["Outcome"] #diabetes/tidak

# Handling missing value
cols_with_zero = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

for col in cols_with_zero:
    X[col] = X[col].replace(0, X[col].median())
print(X.isnull().sum())

df = pd.concat([X, y], axis=1)

# =====================
# EDA
# =====================
df.describe().to_csv("eda_describe.csv")

# histogram
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()

# distribusi target
plt.figure()
sns.countplot(x='Outcome', data=df)
plt.title("Distribusi Target (Diabetes)")
plt.savefig("target_distribution.png")
plt.close()

# correlation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("correlation.png")
plt.close()

# boxplot feature vs target
for col in df.columns[:-1]:
    plt.figure()
    sns.boxplot(x='Outcome', y=col, data=df)
    plt.title(f"{col} vs Outcome")
    plt.savefig(f"boxplot_{col}.png")
    plt.close()

# outlier
outlier_summary = {}

for col in df.columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_summary[col] = len(outliers)
    df[col] = df[col].clip(lower, upper)
    
    outlier_df = pd.DataFrame(
        list(outlier_summary.items()),
        columns=['Feature', 'Outlier_Count']
    )

plt.figure(figsize=(10,6))
sns.barplot(x='Feature', y='Outlier_Count', data=outlier_df)
plt.title("Jumlah Outlier per Feature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outlier_barplot.png")
plt.close()

# skewness
df.skew().to_csv("skewness.csv")
skew = df.skew()

plt.figure(figsize=(10,6))
skew.sort_values().plot(kind='barh')
plt.title("Skewness tiap fitur")
plt.axvline(0, color='red')  
plt.savefig("skewness_plot.png")
plt.close()

# zero values
(df == 0).sum().to_csv("zero_values.csv")

# =====================
# SAVE HASIL PREPROCESS
# =====================
df.to_csv("processed.csv", index=False)

print("Preparation selesai")
