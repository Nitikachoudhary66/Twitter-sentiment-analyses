from flask import Flask, render_template, request
import pickle
import re
from nltk.tokenize import word_tokenize

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
with open('RandomForestClassifier (6).pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Neutral sentiment keywords
neutral_keywords = [
    "okay", "alright", "decent", "average", "indifferent",
    "neither good nor bad", "not sure", "pretty average",
    "nothing special", "meh", "it's fine", "good but bad",
    "great but worst", "good and bad", "not too great", "acceptable",
    "can i play game", "may i play", "should i try", "is it okay", "i can play this game", "like but not so much", " bad but not too much", "i can play game"
    "good baad", "great worst", "fair enough", "not the best",
    "not the worst", "so-so", "kind of okay", "sort of good",
    "better but not great", "not really bad", "could be worse",
    "not too bad", "not too good", "mediocre", "mixed feelings",
    "happy but sad", "excited yet nervous", "good but complicated",
    "balanced emotions", "improved but lacking", "great but flawed", "worst but good", " bad but good",
    "positive but concerned", "negative but hopeful", "good but not bad", "bad but not good", "worst but not so bad"
    "worst but not so bad", "good but not so bad", "worst but not bad", "good but nothing special", 
    "decent", "neither good nor bad", "I don't like this game, but it's fine ", "I don't like it, but it's fine", 
    "average", "okish", "good but not so great", "bad but not so bad", "good but not great", "bad but not bad"
]

# Preprocessing function
def preprocess_text(text, advanced=False):
    # Basic preprocessing
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'^RT\s+', '', text)
    
    # Advanced preprocessing: Handle negations if enabled
    if advanced:
        text = handle_negation(text)
    
    return text

# Handle negations in text
def handle_negation(text):
    tokens = word_tokenize(text)
    negation_words = ["do not", "would not", "could not", "not", "no", "never", "n't", "wouldn't", "shouldn't", "couldn't", "won't", "can't"]
    negated_tokens = []
    negation_flag = False
    negation_phrase = []

    for token in tokens:
        if any(neg in token for neg in negation_words):
            negation_flag = True  # Start tagging the negated part
            negation_phrase = [token]
        elif negation_flag:
            negation_phrase.append(token)
            if token in [".", "!", "?", ";", ","]:  # End of the negated phrase
                negation_flag = False
                negated_tokens.append("NEG_" + " ".join(negation_phrase))
                negation_phrase = []
        else:
            negated_tokens.append(token)

    return " ".join(negated_tokens)

# Function to detect neutral sentiments
def label_neutral(text):
    for word in neutral_keywords:
        if word in text:
            return "neutral"
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input text
    text = request.form['text'].strip()
    
    # Validate input
    if len(text) < 7 or not re.search(r'[a-zA-Z]', text):
        return render_template('index.html', prediction_text="Please enter valid input!")
    
   
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Check if the processed text is still valid
    if len(processed_text.split()) < 2:
        return render_template('index.html', prediction_text="Please enter valid input!")
    
    # Check for neutral sentiment
    neutral_sentiment = label_neutral(processed_text)
    if neutral_sentiment:
        sentiment = neutral_sentiment
    else:
        # Predict sentiment using the model
        prediction = model.predict([processed_text])
        sentiment = prediction[0]
    
    return render_template('index.html', prediction_text=f"Sentiment: {sentiment}")

if __name__ == '__main__':
    app.run(debug=True)
