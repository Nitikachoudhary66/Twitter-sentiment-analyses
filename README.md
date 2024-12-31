# Twitter Sentiment Analysis of Tweets

This project performs sentiment analysis on tweets using machine learning. The goal is to classify tweets as either positive or negative based on their content. It uses a Random Forest Classifier for classification, with a pre-trained model and TF-IDF vectorizer.
## LIVE DEMO 

## Project Structure

The project directory contains the following files:

├── app.py # Main Flask application that serves the model ├── RandomForestClassifier (3).pkl # Pre-trained Random Forest model for sentiment analysis ├── Sentiment Analysis of Tweets live.gif # GIF showing live sentiment analysis ├── Tfidfvectorizer (4).pkl # Pre-trained TF-IDF vectorizer ├── Untitled6 (3).ipynb # Jupyter notebook with model training and analysis (optional) └── templates/ # HTML templates folder (Flask uses this)


### Files Description:
1. **app.py**: The main Python script running a Flask web application. It loads the trained Random Forest model and TF-IDF vectorizer, and exposes an API endpoint to predict tweet sentiment.
2. **RandomForestClassifier (3).pkl**: A serialized machine learning model (Random Forest Classifier) trained to classify tweet sentiment.
3. **Tfidfvectorizer (4).pkl**: A serialized TF-IDF vectorizer used for transforming tweet text data into numerical vectors that the Random Forest model can understand.
4. **Sentiment Analysis of Tweets live.gif**: A GIF showing the live sentiment analysis process.
5. **Untitled6 (3).ipynb**: A Jupyter notebook (optional) containing the code for training the model and performing analysis. This notebook includes the steps for data preprocessing, model training, and evaluation.

## Setup Instructions

1. **Clone the Repository**:  
   Download the project directory or clone the repository to your local machine.

2. **Install Dependencies**:  
   Make sure you have Python installed (preferably version 3.6 or higher). Install the required libraries using pip:
   
   ```bash
   pip install -r requirements.txt
3.Running the Application: Navigate to the project directory in your terminal and run the following command to start the Flask app:

bash
Copy code
python app.py
This will start the Flask server. You can view the application by navigating to http://127.0.0.1:5000/ in your web browser.

Using the Application:
After running the Flask app, you can enter tweet text into the input form on the web page to analyze its sentiment. The application will predict whether the sentiment is positive or negative and display the result on the screen.

How the Sentiment Analysis Works
Input: The user enters a tweet or text in the input box.
Preprocessing: The tweet text is transformed using the pre-trained TF-IDF vectorizer.
Prediction: The transformed text is passed to the Random Forest Classifier model to predict sentiment.
Output: The predicted sentiment (positive/negative) is displayed on the web page.
Model and Preprocessing
The Random Forest Classifier model was trained on a dataset of tweets, and it uses the TF-IDF representation of the text as input for classification.
The TF-IDF vectorizer is used to convert tweet text into numerical vectors, capturing important words and their significance based on term frequency and inverse document frequency.
Additional Notes
The project is built using Flask, a lightweight web framework for Python, to deploy the model as a web service.
You can modify the model or retrain it with your own dataset by using the Untitled6 (3).ipynb Jupyter notebook.
