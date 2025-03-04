from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Sample pipeline with TF-IDF and Random Forest
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', RandomForestClassifier())
])

# Hyperparameter tuning
param_grid = {
    'tfidf__max_df': [0.75, 1.0],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20]
}

# Grid search for best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Evaluate on test set
y_pred = grid_search.predict(X_test)
print("Test set evaluation: ")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
