import joblib
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the saved SVM model and TfidfVectorizer
model = joblib.load('models/svm_model_reg_1.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Define a secret key for validation
SECRET_KEY = 'Window is kinda cringe:3'


@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    if request.method == 'POST':
        tweet = request.form['tweet']
        key = request.form['key']

        if key == SECRET_KEY:
            # Transform the tweet using the loaded TfidfVectorizer
            transformed_tweet = tfidf_vectorizer.transform([tweet])
            sentiment = model.predict(transformed_tweet)[0]
            if sentiment == 1:
                sentiment = 'Positive'
            else:
                sentiment = 'Negative'
            message = f'Sentiment: {sentiment}'
        else:
            message = 'Invalid key!'

    return render_template('index.html', message=message)


if __name__ == '__main__':
    app.run(debug=True)
