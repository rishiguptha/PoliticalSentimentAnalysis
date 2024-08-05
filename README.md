# Political Sentiment Analysis and Election Outcome Forecasting

This project aims to analyze political sentiment from Twitter data and forecast election outcomes using logistic regression and strategic feature selection.

## Project Overview

The project involves the following steps:
1. Data Collection and Preprocessing
2. Sentiment Analysis
3. Feature Selection
4. Building and Training a Logistic Regression Model
5. Predicting the Election Outcome

## Data

The initial dataset consists of tweets from political figures, with the following columns:
- `Party`: The political affiliation of the tweet author (`Democrat` or `Republican`).
- `Handle`: The Twitter handle of the tweet author.
- `Tweet`: The original tweet text.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/PoliticalSentimentAnalysis.git
    cd PoliticalSentimentAnalysis
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Preprocessing

1. Load the dataset:
    ```python
    import pandas as pd
    tweets = pd.read_csv('path_to_your_data.csv')
    ```

2. Clean the tweets and compute sentiment scores:
    ```python
    from textblob import TextBlob
    import re

    # Function to clean tweet text
    def clean_tweet(text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#', '', text)  # Remove hashtags
        text = re.sub(r'RT', '', text)  # Remove retweet indicator
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        return text

    # Apply the cleaning function
    tweets['CleanedTweet'] = tweets['Tweet'].apply(clean_tweet)

    # Compute subjectivity and polarity
    tweets['Subjectivity'] = tweets['CleanedTweet'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    tweets['Polarity'] = tweets['CleanedTweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Classify based on subjectivity and polarity
    tweets['Classify_S'] = tweets['Subjectivity'].apply(lambda x: 'Subjective' if x > 0.5 else 'Not Subjective')
    tweets['Classify_P'] = tweets['Polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    ```

### Modeling

1. Select features and target:
    ```python
    X = tweets[['Subjectivity', 'Polarity']]  # You can add more features here
    y = tweets['Party']

    # Encode the target variable
    y = y.map({'Democrat': 0, 'Republican': 1})  # Adjust according to your data
    ```

2. Train-test split and feature selection:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFE

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature selection using RFE
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select=2)  # Adjust the number of features as needed
    fit = rfe.fit(X_train, y_train)

    # Selecting the top features
    selected_features = X_train.columns[fit.support_]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    ```

3. Build and train the logistic regression model:
    ```python
    logreg = LogisticRegression()
    logreg.fit(X_train_selected, y_train)

    # Make predictions
    y_pred = logreg.predict(X_test_selected)

    # Evaluate the model
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    ```

4. Predict the winning party:
    ```python
    # Predict the probabilities on the test set
    probabilities = logreg.predict_proba(X_test_selected)

    # Get the predicted party (0 for Democrat, 1 for Republican)
    predicted_party = logreg.predict(X_test_selected)

    # Determine the overall predicted winner
    # Count the number of each predicted class
    predicted_counts = np.bincount(predicted_party)

    # The party with the higher count is predicted to win
    predicted_winner = np.argmax(predicted_counts)

    # Map back to party names
    party_map = {0: 'Democrat', 1: 'Republican'}
    print(f"The predicted winning party is: {party_map[predicted_winner]}")
    ```

## Results

The logistic regression model predicts the winning party based on the sentiment analysis of the tweets. The results are evaluated using accuracy, confusion matrix, and classification report.

## Interactive Features

You can interact with the project by modifying the `path_to_your_data.csv` to your dataset path. You can also add more features to the model and experiment with different feature selection methods.

### Example Interactive Code Block

```python
# Modify and run this block to experiment with different features
new_features = ['Subjectivity', 'Polarity', 'LengthOfTweet']
tweets['LengthOfTweet'] = tweets['CleanedTweet'].apply(len)

X = tweets[new_features]
y = tweets['Party'].map({'Democrat': 0, 'Republican': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Accuracy with new features:", accuracy_score(y_test, y_pred))
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
