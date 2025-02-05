import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from dotenv import load_dotenv
from datasets import load_dataset
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql').to(device)
model.eval()

embedding_model = SentenceTransformer('all-mpnet-base-v2')

dataset = load_dataset("b-mc2/sql-create-context", split="train")

embedding_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)

sql_texts = [example['question'] for example in dataset]
sql_embeddings = embedding_model.encode(sql_texts, convert_to_numpy=True)
index.add(sql_embeddings)


def retrieve_similar_queries(user_query, top_k=3):
    query_embedding = embedding_model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [sql_texts[i] for i in indices[0]]


def generate_sql(input_prompt):
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, temperature=0, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("Simple RAG SQL Query Generator'a hoş geldiniz!")

    while True:
        query = input("SQL oluşturmak için bir doğal dil sorgusu girin (çıkmak için 'exit'): ")
        if query.lower() == "exit":
            break

        # En benzer SQL sorgularını getir
        retrieved_queries = retrieve_similar_queries(query)
        context = "\n".join(retrieved_queries)

        # Model girdisi olarak FAISS'ten alınan bağlamı ekle
        input_prompt = f"tables:\n{context}\nquery for: {query}"

        generated_sql = generate_sql(input_prompt)
        print(f"Oluşturulan SQL sorgusu: {generated_sql}\n")


if __name__ == "__main__":
    main()
