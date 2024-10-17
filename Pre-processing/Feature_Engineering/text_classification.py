from src.utils import load_volunteer_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


volunteer = load_volunteer_dataset()
volunteer.dropna(subset = ['category_desc'], inplace = True)
# Take the title text
title_text = volunteer["title"]
# Create the vectorizer method
tfidf_vec = TfidfVectorizer()
# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)
# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"]
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify=y, random_state=42)

nb = GaussianNB()
# Fit the model to the training data
nb.fit(X_train, y_train)
# Print out the model's accuracy
print(nb.score(X_test, y_test))