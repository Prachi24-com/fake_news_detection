# Fake News Detection

## Overview
This project aims to detect fake news articles using machine learning and natural language processing (NLP) techniques. The model is trained to classify news articles as either "Fake" or "Real" based on textual content.

## Features
- Data preprocessing (tokenization, stopword removal, stemming, etc.)
- TF-IDF vectorization for feature extraction
- Machine learning models (Logistic Regression, Random Forest, Naive Bayes, etc.)
- Deep learning model implementation using LSTM
- Model evaluation with accuracy, precision, recall, and F1-score
- Web application interface for real-time fake news detection

## Dataset
The dataset used for training and evaluation is sourced from the **Kaggle Fake News Dataset**, which consists of labeled real and fake news articles.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow/Keras (for deep learning model)
- NLTK and spaCy (for NLP preprocessing)
- Flask (for web deployment)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download necessary NLP resources:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage
1. Train the model:
   ```bash
   python train.py
   ```
2. Test the model:
   ```bash
   python test.py
   ```
3. Run the web application:
   ```bash
   python app.py
   ```
4. Access the web app at `http://127.0.0.1:5000/`

## Results
The model achieved an accuracy of **99%** on the test dataset. The LSTM-based deep learning model showed improved performance compared to traditional ML models.

## Future Enhancements
- Integration of real-time news article analysis
- Implementation of transformers (BERT, RoBERTa) for improved classification
- Deployment as a cloud-based service

## Contributors
- Prachi Srivastava ([GitHub Profile](https://github.com/Prachi24-com))

## License
This project is licensed under the MIT License.

