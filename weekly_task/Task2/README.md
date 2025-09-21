# Task 2: NLTK-Powered Text Analytics Web App

## Description
This task implements a web-based text analytics tool using **NLTK**, **Flask**, and **pandas**.
Users can upload a text file, which is then cleaned, tokenized, analyzed, and visualized.

## Features
- Text preprocessing (tokenization, stopwords, lemmatization)
- Frequency distribution of words
- Sentiment analysis using VADER
- Interactive web interface with file upload
- Visualizations of top word frequencies

## Files
- `app.py`: Flask web app
- `nlp_pipeline.py`: NLP preprocessing and analysis functions
- `templates/`: HTML templates for UI
- `example_data.txt`: Sample text file
- `README.md`: Instructions

## Run Instructions
1. Install dependencies:
   ```bash
   pip install flask nltk pandas matplotlib
