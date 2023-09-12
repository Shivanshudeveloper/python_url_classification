import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib as jbl
import pickle
labels = {0: 'Adult', 1: 'Business/Corporate', 2: 'Computers and Technology', 3: 'E-Commerce', 4: 'Education', 5: 'Food', 6: 'Forums', 7: 'Games', 8: 'Health and Fitness', 9: 'Law and Government', 10: 'News', 11: 'Photography', 12: 'Social Networking and Messaging', 13: 'Sports', 14: 'Streaming Services', 15: 'Travel'}
data = pd.read_csv("website_classification.csv")
x = data.cleaned_website_text
y = data.Category
print(y.unique())
lbe = LabelEncoder()
lbe.fit(y)
y = lbe.transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
model = Pipeline([
    ('vecorizer', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
model.fit(X_train, y_train)
a=model.score(X_test, y_test)
print(a)

# jbl.dump(model, 'model.pkl')
# with open("model2.pkl","wb") as f:
#     pickle.dump(model,f)
# ind=model.predict(["playing football"])
# print(labels.get(ind[0]))