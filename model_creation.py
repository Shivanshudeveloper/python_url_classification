import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your dataset (replace 'your_data.csv' with your CSV file)
df = pd.read_csv('categories.csv')

# Preprocessing: Combine domain and title into a single text_data column
df['text_data'] = df['title'] + ' ' + df['domain']

# Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['text_data'])  # Fit on the entire dataset

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Choose a classification model (e.g., Multinomial Naive Bayes)
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'website_classifier_model.joblib')
print("Successfully trained model")
