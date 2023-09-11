import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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
# print(model.predict(["education"]))