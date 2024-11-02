import streamlit as st
import pandas as pd
import pymysql
from sentence_transformers import SentenceTransformer, util  # Importing util for cosine similarity
import numpy as np

# Connect to the MySQL database
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='abcd',
    database='test'
)
cursor = conn.cursor()

st.title("Quora Question Pair Similarity")

# Load RoBERTa model
model = SentenceTransformer('roberta-base-nli-mean-tokens')

# Take input from user
question = st.text_input("Enter a Question")

# Set similarity threshold
similarity_threshold = 0.7  # Adjust this threshold as needed

if question:
    # Preprocess the input question
    # Clean, tokenize, etc., as per your preprocessing steps
    
    # Encode the input question using RoBERTa
    input_vector = model.encode(question)
    
    # Fetch questions from the database
    cursor.execute("SELECT id, question FROM questions")  
    rows = cursor.fetchall()
    
    # Compute similarities between the input question and questions in the database
    similarities = []
    for row in rows:
        question_id, db_question = row
        db_vector = model.encode(db_question)
        cosine_sim = util.cos_sim(input_vector, db_vector)[0][0]  # Using util.cos_sim for cosine similarity
        
        # Filter out non-similar questions based on the threshold
        if cosine_sim >= similarity_threshold:
            similarities.append((question_id, db_question, cosine_sim))
    
    # Sort the similarities
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Display similar questions or message if no similar questions found
    if similarities:
        st.subheader("Similar Questions:")
        for sim_question in similarities[:5]:  # Display top 5 similar questions
            # Display question and relevance percentage
            st.write(f"- **{sim_question[1]}** (Relevance: {sim_question[2]*100:.2f}%)")
    else:
        st.error("Oops! No similar questions found.")

# Close the database connection
conn.close()
