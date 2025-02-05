import torch
from datasets import load_dataset
from transformers import T5Tokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import re
from tqdm import tqdm
import csv
import pandas as pd
import os

device = torch.device("mps")
#print(torch.backends.mps.is_available())

dataset = load_dataset("b-mc2/sql-create-context")
dataset = dataset.filter(
    lambda x: x['question'].strip() != '' and x['context'].strip() != '' and x['answer'].strip() != ''
)

train_data = dataset["train"]
print(dataset.keys())
# print(dataset_sample)

tokenizer = T5Tokenizer.from_pretrained("t5-small")


def tokenize_text(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")


model_name = "cssupport/t5-small-awesome-text-to-sql"
llm_pipeline = pipeline(
    "text2text-generation",
    model=model_name,
    device=device,
    max_length=512,
    temperature=0.0,
    do_sample=False
    #    return_tensors="pt" ---> generated_text key erroru alındığı için kaldırdım, hata giderildi
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def get_embeddings(texts):
    return embedding_model.embed_documents(texts)


SPECIAL_SEPARATOR = "\n---\n"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=5,
    separators=[SPECIAL_SEPARATOR]
)


def save_embeddings_to_csv(embeddings, file_name='embeddings.csv'):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["embedding_id"] + [f"dim_{i}" for i in range(len(embeddings[0]))])  # Başlıklar
        for i, embedding in enumerate(embeddings):
            writer.writerow([i] + np.array(embedding).tolist())  # NumPy array'e çevirip listeye dönüştür


def save_chunks_to_csv(documents, file_name='chunks.csv'):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["chunk_id", "chunk_text"])  # Başlıklar
        for i, chunk in enumerate(documents):
            writer.writerow([i, chunk])


embeddings_file = 'embeddings.csv'
chunks_file = 'chunks.csv'

if os.path.exists(embeddings_file) and os.path.exists(chunks_file):
    print("Önceden kaydedilmiş dosyalar bulundu. Direkt yüklenecek...")

    chunks_df = pd.read_csv(chunks_file)
    embeddings_df = pd.read_csv(embeddings_file)

    documents = chunks_df['chunk_text'].tolist()
    embeddings = embeddings_df.drop(columns=["embedding_id"]).values.tolist()
else:
    print("Önceden kaydedilmiş dosyalar bulunamadı. Chunk'lama ve embedding işlemi başlatılıyor...")

    documents = []
    for q, c, a in tqdm(zip(train_data["question"], train_data["context"], train_data["answer"]),
                        total=len(train_data["answer"]), desc="Processing"):
        full_text = f"Question: {q}\nContext: {c}\nAnswer: {a}"
        chunks = text_splitter.split_text(full_text)
        documents.extend(chunks)

    print(f"Yeni toplam chunk sayısı: {len(documents)}")

    embeddings = []
    for chunk in tqdm(documents, desc="Processing embeddings", total=len(documents)):
        chunk_embedding = get_embeddings([chunk])
        embeddings.append(chunk_embedding[0])

    print(f"Embedding işlemi tamamlandı, toplam embedding sayısı: {len(embeddings)}")

    # Verileri CSV'ye kaydet
    save_embeddings_to_csv(embeddings, embeddings_file)
    save_chunks_to_csv(documents, chunks_file)

    print("Embedding'ler ve chunk'lar başarıyla CSV dosyasına kaydedildi.")

# FAISS vektör veritabanını oluşturma
dim = len(embeddings[0])  # Her embedding'in boyutu
index = faiss.IndexFlatL2(dim)  # L2 normlu düz bir index

faiss_index = np.array(embeddings)
index.add(faiss_index)

print(f"FAISS index'e toplam {index.ntotal} embedding eklendi.")


def search_similar_questions(query, index, k=3):
    # query_embedding = get_embeddings([query])
    query_embedding = np.array(get_embeddings([query]), dtype=np.float32)
    D, I = index.search(np.array(query_embedding), k)

    return D, I


def extract_number(text):
    """Metindeki ilk sayıyı döndürür, yoksa None döner."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


"""
def detect_operator(text):
    if "older than" in text:
        return ">"
    elif "younger than" in text:
        return "<"
    return None
"""


def extract_state_and_name(text):
    """Kullanıcının girdisinden eyalet ve özel isimleri çıkarır."""
    state_pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b'  # Büyük harfle başlayan kelimeleri bulur (eyalet ve isimler için)
    name_pattern = r'\b[A-Z][a-z]+\b'  # Özel isimleri tanımlar

    possible_matches = re.findall(state_pattern, text)
    possible_names = re.findall(name_pattern, text)

    return possible_matches, possible_names


def update_sql_query(sql_query, user_query):
    """SQL sorgusundaki aralık değerlerini (BETWEEN x AND y) günceller."""
    user_numbers = re.findall(r'\d+', user_query)  # Kullanıcı girdisindeki tüm sayıları al
    sql_numbers = re.findall(r'\d+', sql_query)  # SQL sorgusundaki tüm sayıları al

    if len(user_numbers) == 2 and "BETWEEN" in sql_query:
        # Eğer SQL sorgusunda BETWEEN x AND y varsa ve kullanıcıdan iki sayı geldiyse
        updated_query = re.sub(r'BETWEEN \d+ AND \d+', f'BETWEEN {user_numbers[0]} AND {user_numbers[1]}', sql_query)
        return updated_query
    elif len(user_numbers) == 1 and sql_numbers:
        # Eğer sadece tek bir sayı varsa, ilk bulunan sayıyı değiştir
        updated_query = sql_query.replace(sql_numbers[0], user_numbers[0])
        return updated_query

    """SQL sorgusundaki değerleri (örn: eyalet ismi) günceller."""
    # Kullanıcı sorgusundaki eyalet ismini bul
    state_match = re.search(r"state\s*'([^']+)'", user_query)
    if state_match:
        user_state = state_match.group(1)

        # SQL sorgusundaki mevcut state değerini bul
        sql_state_match = re.search(r"born_state\s*=\s*'([^']+)'", sql_query)
        if sql_state_match:
            sql_state = sql_state_match.group(1)

            # Eğer SQL'deki state ile user'dan gelen farklıysa güncelle
            if user_state != sql_state:
                updated_query = sql_query.replace(f"'{sql_state}'", f"'{user_state}'")
                return updated_query

    return sql_query  # Eğer değişiklik gerekmiyorsa, orijinal SQL sorgusunu döndür


def generate_answer(query, index, model, k=3):
    # query_parts = re.split(r'\band\b|\balso\b|\.|\?', query, flags=re.IGNORECASE)
    # query_parts= re.split(r'\b(?:and|also|And|Also|\?|\. )\b', user_input)
    query_parts = re.split(r'(?<=[?.!])\s+', query.strip())  # Cümle sonlarına göre ayır
    query_parts = [q.strip() for q in query_parts if q.strip()]  # Boşlukları temizle

    sql_queries = []

    for part in query_parts:

        print(f"\nProcessing query part: {part}")

        # Benzerlik araması yap
        # distances, indices = search_similar_questions(query, index, k)
        distances, indices = search_similar_questions(part, index, k)

        print(f"FAISS Matched Indices: {indices}")
        print(f"FAISS Distances: {distances}")

        similar_questions = []
        for i in range(k):
            # idx = indices[0][i]
            idx = int(indices[0][i])
            similar_questions.append({
                "question": train_data['question'][idx],
                "context": train_data['context'][idx],
                "answer": train_data['answer'][idx]
            })
        print(similar_questions)

        similar_example = train_data[int(indices[0][0])]  # En iyi eşleşmeyi al
        matched_query = similar_example["question"]
        matched_sql = similar_example["answer"]

        print(f"Matched Question: {matched_query}")
        print(f"Matched SQL: {matched_sql}")

        updated_sql = update_sql_query(matched_sql, part)
        sql_queries.append(updated_sql)

    return sql_queries


if __name__ == "__main__":
    while True:
        user_input = input("Lütfen SQL sorgusu oluşturmak için bir soru girin (Çıkmak için 'exit' yazın): ")
        prompt_template = f"""
                The SQL query generator should always use logical reasoning.

                Examples:
                - "Older than" means ">" in SQL.
                - "Younger than" means "<" in SQL.
                - "Before 2000" means "< 2000" in SQL.
                - "After 1990" means "> 1990" in SQL.
                - "Equal to 50" means "= 50" in SQL.

                If the user asks a meaning-related question, answer with a definition.
                If the user asks for a SQL query, generate the SQL.


                Here is a natural language question:
                "{user_input}"

                """
        if user_input.lower() in ["exit", "çıkış"]:
            print("Programdan çıkılıyor...")
            break

        responses = generate_answer(user_input, index, llm)

        for i, response in enumerate(responses, 1):
            print(f"Response {i}: {response}")


