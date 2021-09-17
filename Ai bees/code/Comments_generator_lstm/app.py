from flask import Flask, render_template, url_for, request
import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    client = pymongo.MongoClient("mongodb://localhost:27017/")

    db = client['taimoor_text_generation_db']
    collection = db['taimoorsamplecollection']

    from joblib import load, dump
    pipeline = load("text_classification.joblib")
    if request.method == 'POST':
        message = request.form['message']
        data = [message]

        my_prediction = pipeline.predict(data)
        datadb = {'Post': message, 'Category': my_prediction}
        collection.insert_one(datadb)

    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)