
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

# Load the movie review data
movie_rev = pd.read_csv("movie_reviews.csv")

print(movie_rev.head)
print(movie_rev.describe(include="all"))
print(movie_rev.columns)

na=movie_rev.isnull().sum()
print(na)

# Distribution of sentiments
plt.figure(figsize=(6, 4))
sentiment_counts = movie_rev['Sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(range(len(sentiment_counts)), ['Negative', 'Positive'], rotation=0)
plt.show()


# Visualize the most frequent words in the reviews
count_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = count_vectorizer.fit_transform(movie_rev['Review'])
words = count_vectorizer.get_feature_names_out()
print(words)
word_frequencies = X.sum(axis=0).A1
print(word_frequencies)
word_freq_dict = dict(zip(words, word_frequencies))
sorted_word_freq = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)[:25]
print(word_freq_dict)


plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in sorted_word_freq], [x[1] for x in sorted_word_freq], color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Top 20 Most Frequent Words in Reviews')
plt.gca().invert_yaxis()
plt.show()


# Analyze the distribution of review lengths
review_lengths = movie_rev['Review'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
sns.histplot(review_lengths, bins=30, color='purple', edgecolor='black')
plt.title('Distribution of Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.show()

