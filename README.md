# Question Similarity Finder with BERT and MySQL

This project is a question similarity finder application built with a MySQL database to store questions and BERT-based embeddings to find similar questions. Given an input question, the application retrieves the most similar questions from the database by calculating cosine similarity between BERT embeddings.

In addition to the BERT model, the project explores other machine learning models, including MaLSTM, XGBoost, and Random Forest, for training and validating question similarity results. The Streamlit frontend provides a user-friendly interface for inputting questions and displaying similar questions with relevance scores.

**Features:**
- **MySQL Database**: Stores a large collection of questions for similarity matching.
- **BERT Model**: Encodes questions and calculates similarity using cosine similarity.
- **Alternative Models**: Experiments with MaLSTM, XGBoost, and Random Forest for comparison and validation.
- **Streamlit Interface**: Interactive UI for entering questions and viewing top similar results.

This project combines NLP and database management to streamline similarity matching for question-based datasets.
