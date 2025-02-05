import streamlit as st
from sql_generator import generate_answer, index, llm

st.title("SQL Query Generator")
st.write("Enter a natural language query, and the system will generate an SQL query.")

user_input = st.text_input("Enter your query:")

if user_input:
    st.write("Processing...")
    responses = generate_answer(user_input, index, llm)

    for i, response in enumerate(responses, 1):
        st.code(f"Response {i}: {response}")
