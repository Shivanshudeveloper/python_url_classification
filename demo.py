import pickle
labels = {0: 'Adult', 1: 'Business/Corporate', 2: 'Computers and Technology', 3: 'E-Commerce', 4: 'Education', 5: 'Food', 6: 'Forums', 7: 'Games', 8: 'Health and Fitness', 9: 'Law and Government', 10: 'News', 11: 'Photography', 12: 'Social Networking and Messaging', 13: 'Sports', 14: 'Streaming Services', 15: 'Travel'}
with open("model2.pkl","rb") as f:
    mf=pickle.load(f)
a=mf.predict(["play football"])
print(labels.get(a[0]))