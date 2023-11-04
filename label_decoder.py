import joblib

# Load the saved label encoder
label_encoder = joblib.load('label_encoder.joblib')
vectorizer = joblib.load('tfif_vectorizer.joblib')
model = joblib.load('website_classifier_model.joblib')

# original = vectorizer.transform(["Trello"])
original = vectorizer.transform(["Academic Tutorials"])
# original = vectorizer.transform(["Whatsapp"])
original = vectorizer.transform(["Slack"])
# original = vectorizer.transform(["Valorant"])
# original = vectorizer.transform(["Figma"])
original_word_vector = vectorizer.inverse_transform(original)

# print(original,"vectorizer word")
# print(original_word_vector[0],"vectorizer")

# encoded_label = label_encoder.transform(['Productivity Tools'])  # Returns [1]
# Encode a label

# print(encoded_label)
result = model.predict(original)
result_word = label_encoder.inverse_transform(result)
print(result)
print(result_word)
# print(original[0],"original")
# Inverse transform to get the original 

# original_label = label_encoder.inverse_transform(encoded_label)  # Returns ['label2']

# print(original_label)

