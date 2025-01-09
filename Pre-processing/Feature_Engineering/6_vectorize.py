from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Take the title text
title_text = volunteer["title"]

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)

print(text_tfidf)