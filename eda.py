import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("news_data.csv")

print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df["category"].value_counts().head(10))
print("\nUnique Categories:")
print(df["category"].nunique())
print("\nTop 20 Categories:")
print(df["category"].value_counts().head(20))
df["category"].value_counts().head(10).plot(
    kind="bar",
    figsize=(10,5)
)

plt.title("Top 10 News Categories")
plt.tight_layout()

plt.show()
