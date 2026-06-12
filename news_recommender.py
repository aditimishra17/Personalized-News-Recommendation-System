import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("news_data.csv")

# For testing
df = df.head(1000)

# Create vectors
cv = CountVectorizer(stop_words="english")

count_matrix = cv.fit_transform(df["comb"])

# Similarity matrix
similarity = cosine_similarity(count_matrix)

# Sample article
title = df["headline"].iloc[0]

print("Selected Article:")
print(title)

print("\nRecommended Articles:\n")

idx = df[df["headline"] == title].index[0]

scores = list(enumerate(similarity[idx]))

scores = sorted(scores, key=lambda x: x[1], reverse=True)

for i in scores[1:6]:
    print(df.iloc[i[0]]["headline"])