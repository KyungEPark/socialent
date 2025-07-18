## Using CorEx for topic modeling
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd
from util import *

# Call text corpora (to be updated after the text corpora is ready)
examples = [
    "Ich bin ein Berliner.",
    "Ich habe seit ich jung war, immer gerne Fußball gespielt.",
    "Ich bin damit verantwortlich, Umwelt zu schutzen.",
    "Wir sind profitabel aber auch menschenfreundlich.",
    "Atemschutz ist ein Menschenrecht."
]

# Call list of keywords
keywords = call_csv('data/keywordlist.csv')
print(keywords.head())

# Run preprocessing
category_keywords = preprocess_keywords(keywords)

# Vectorize the texts
german_stopwords = stopwords.words('german')
vectorizer = CountVectorizer(stop_words=german_stopwords)
X = vectorizer.fit_transform(examples)
vocab = vectorizer.get_feature_names_out()

# Anchor words: filter to those that actually exist in vocab
anchor_words = []
category_names = []

for category, keywords in category_keywords.items():
    anchors = [kw for kw in keywords if kw in vocab]
    if anchors:
        anchor_words.append(anchors)
        category_names.append(category)

# Fit the CorEx model
topic_model = ct.Corex(n_hidden=len(anchor_words), words=vocab, seed=42)
topic_model.fit(X, words=vocab, anchors=anchor_words, anchor_strength=3)

# Predict topic activation per document
labels = topic_model.predict(X)
df_corex = pd.DataFrame(labels, columns=category_names)
df_corex['text'] = examples

# Save to CSV
df_corex.to_csv('data/corex_topics.csv', index=False)
print(df_corex)

# Save total correlation scores
tc_df = pd.DataFrame({
    'Category': category_names,
    'TotalCorrelation': topic_model.tcs
})
tc_df.to_csv('data/corex_total_correlation.csv', index=False)

