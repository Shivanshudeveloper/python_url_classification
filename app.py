# app.py
from flask import Flask, request, jsonify
import difflib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker 
import pickle
labels = {0: 'Adult', 1: 'Business/Corporate', 2: 'Computers and Technology', 3: 'E-Commerce', 4: 'Education', 5: 'Food', 6: 'Forums', 7: 'Games', 8: 'Health and Fitness', 9: 'Law and Government', 10: 'News', 11: 'Photography', 12: 'Social Networking and Messaging', 13: 'Sports', 14: 'Streaming Services', 15: 'Travel'}
data = pd.read_csv("website_classification.csv")

app = Flask(__name__)

# MongoDB configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/projectracker'
mongo = MongoClient(app.config['MONGO_URI'])
db = mongo.projectracker
engine = create_engine('postgresql://tsdbadmin:nnbndcdsnnkfemu5@g15kroocga.ied5mx5496.tsdb.cloud.timescale.com:33332/tsdb')
Session = sessionmaker(bind=engine)

def create_data_match():
    # Create a new session
    session = Session()
    try:
        # Define the SQL query to create the img_data table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS data_match (
            img_id VARCHAR(255),
            category TEXT
        )
        """
        
        # Execute the query to create the table
        session.execute(text(create_table_query))
        
        # Commit the changes to the database
        print("success\n")
        session.commit()
    except Exception as e:
        # Roll back the transaction if an error occurs
        session.rollback()
        raise e
    finally:
        # Close the session
        session.close()

def insert_img_data(img_id, category):
    # Create a new session
    session = Session()
    try:
        # Define the SQL query to insert data into the table
        insert_query = text("INSERT INTO data_match (img_id, category) VALUES (:img_id, :category)")
        
        # Execute the query with the provided parameters
        session.execute(insert_query, {"img_id": img_id, "category": category})
        
        # Commit the changes to the database
        session.commit()
        print("data inserted successfully")
    except Exception as e:
        # Roll back the transaction if an error occurs
        session.rollback()
        raise e
    finally:
        # Close the session
        session.close()

def decision_tree_classification():
    # Load the Iris dataset (you can replace it with your own dataset)
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree classifier
    clf = DecisionTreeClassifier()

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    return clf, accuracy

def check_words_in_csv3(array_words, csv_file, userId):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Convert DataFrame column to a set for faster word matching
    csv_words_set = set(df['Words'].str.strip().values)

    # Define an empty list to store word match results
    word_match_results = []

    # Iterate through each word from the array
    for word in array_words:
        word = word.strip() # Remove leading/trailing whitespace

        # Initialize variables to keep track of the best match and its similarity ratio
        best_match = None
        best_similarity_ratio = 0.0

        # Calculate the similarity ratio with each CSV word
        for csv_word in csv_words_set:
            csv_word = csv_word.strip() # Remove leading/trailing whitespace
            # print(csv_word)
            # Use SequenceMatcher to calculate the similarity ratio
            matcher = difflib.SequenceMatcher(None, word, csv_word)
            similarity_ratio = matcher.ratio()
            # print(similarity_ratio)
            # Check if the current word has a higher similarity ratio
            if similarity_ratio > best_similarity_ratio:
                best_similarity_ratio = similarity_ratio
                best_match = csv_word

        # Define a threshold for considering a match
        similarity_threshold = 0.2 # You can adjust this threshold as needed

        # Check if the best match meets the similarity threshold
        if best_similarity_ratio >= similarity_threshold:
            # Calculate the probability (similarity ratio as a percentage)
            probability_percentage = best_similarity_ratio * 100
            # print(best_similarity_ratio," is greater than ",similarity_threshold)
            # Append the match result to the list
            word_match_results.append({
                'word': word,
                'project_match': best_match,
                'prob':round(probability_percentage,2),
                'probability': f'{probability_percentage:.2f}%',
                'userId': userId
            })

    # Print or save the word match results as needed
    # print(word_match_results)
    avg_probability_percentage=0
    for match_result in word_match_results:
        avg_probability_percentage+=match_result['prob']
        # print(f"Word from Array: {match_result['word']}, "
        #       f"Matched Word to projects: {match_result['project_match']} "
        #       f"with Probability: {match_result['probability']}")
    avg_probability_percentage=avg_probability_percentage/len(word_match_results)
    print("average_percentage: ",avg_probability_percentage)
    insert_img_data(userId,avg_probability_percentage)
        
def predictCategory(array_words, userId):
    final_string = " ".join(array_words)
    with open("model2.pkl","rb") as f:
        mf=pickle.load(f)
    a=mf.predict([final_string])
    print(labels.get(a[0]))
    
    insert_img_data(userId,labels.get(a[0]))
    
    


def check_words_in_csv(array_words, csv_file, userId):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    # nlp = spacy.load("en_core_web_md")
    # matching_results = []

    # Convert DataFrame column to a set for faster word matching
    csv_words_set = set(df['Words'].str.strip().values)

    # print(csv_words_set)

    # Define a lower threshold for similarity ratio (60% matching in this case)
    similarity_threshold = 0.2

    # Check each word from the array against the CSV words and print if there is a match
    for word in array_words:
        # word_embedding = nlp(word)
        # Find the most similar word(s) in the CSV using difflib
        matches = difflib.get_close_matches(word.strip(), csv_words_set, n=1, cutoff=similarity_threshold)

        # If there is a match, print the word from the array and the matched word from the CSV
        if matches:
            # print("matches",matches)
            matched_word = matches[0]
            # Calculate cosine similarity between the original word and the matched word
            # original_word_vector = np.array([1 if w in word else 0 for w in matched_word.split()])
            # matched_word_vector = np.array([1 if w in matched_word else 0 for w in word.split()])
            # similarity_score = cosine_similarity([original_word_vector], [matched_word_vector])[0][0]

            original_word_vector = np.array([1 if w in word else 0 for w in csv_words_set])
            matched_word_vector = np.array([1 if w in matched_word else 0 for w in csv_words_set])

            # Calculate cosine similarity between the two vectors
            similarity_score = cosine_similarity([original_word_vector], [matched_word_vector])[0][0]
            new_data = {
                'word': word,
                'project_match': matches[0],
                'probability': f'{similarity_score * 100:.2f}%',
                'userId': userId
            }

            # Save the post to MongoDB collection
            # db.projects_matches.insert_one(new_data)
            insert_img_data(userId,new_data['probability'])
            print(f"Word from Array: {word}, Matched Word to projects: {matches[0]} at below {new_data['probability']}")

@app.route('/')
def hello():
    return 'Hello, Flask!'


@app.route('/mlwork', methods=['POST'])
def post_data():
    data = request.json

    # Check if 'data' contains the required keys
    if 'userData' in data and 'userId' in data:
        userData = data['userData']
        userId = data['userId']

        # CSV file path containing words in a single column named 'Words'
        csv_file = "projects.csv"



        # Process the data as needed
        # For this example, we'll just return the received data as JSON
        response_data = {
            'userData': userData,
            'userId': userId
        }
        # userData=userData.split(" ")
        # print(userData)
        # Call the function to check words in the CSV file
        # check_words_in_csv3(userData, csv_file, userId)
        predictCategory(userData, userId)

        return jsonify(response_data), 200
    else:
        return jsonify({'error': 'Invalid JSON data. "my_array" and "my_string" keys are required.'}), 400



if __name__ == '__main__':
    # create_data_match()
    app.run(port=8080, debug=True)
