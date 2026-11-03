import os
from getpass import getpass
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import pyttsx3
from tkinter import Tk, filedialog
from dotenv import load_dotenv
import json
from pathlib import Path
import pickle
import hashlib

load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY couldn't be found at .env file.")
client = Groq(api_key=api_key)
AI_model = "openai/gpt-oss-120b"
Temperature = 0.5
Max_Tokens = 500
storage = Path(__file__).resolve().parent / "rag_storage"
storage.mkdir(exist_ok=True)
chunks_path = storage / "chunks.pkl"
embeddings_path = storage / "embeddings.npy"
index_path = storage / "index.faiss"
metadata_path = storage / "metadata.json"

#reading AI persona
persona = ""
with open("AI_Persona.txt", "r", encoding="utf-8") as content:
    persona = content.read()

#reading data base
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select the file you want to upload.", filetypes=[("Text files", "*.txt")])
root.destroy()
if not file_path:
    print("No file selected or found.")
    exit()
with open(file_path, "r", encoding="utf-8") as data:
    data_base = data.read()
print("Database successfully loaded!")
data_hash = hashlib.sha256(data_base.encode("utf-8")).hexdigest()
embeddings_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cheking storage for info
def check_storage():
    files_exist = (chunks_path.exists() and embeddings_path.exists() and index_path.exists( ) and metadata_path.exists())
    if not files_exist:
        return False

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata["data_hash"] != data_hash:
        return False
    if metadata["embeddings_model"] != embeddings_model_name:  #making sure that the model is the same
        return False

    return True

# Saving the knowledge
def save_knowledge(chunks, embeddings, index):
    with open(chunks_path, "wb") as file:
        pickle.dump(chunks, file)

    np.save(embeddings_path, embeddings)

    faiss.write_index(index, str(index_path))

    metadata = {
        "data_hash": data_hash,
        "embeddings_model": embeddings_model_name
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("Knowledge successfully saved!")

# Chunks
def creating_chunks(text, size=1000, overlay=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlay
    return chunks
chunks = creating_chunks(data_base)

# Embeddings
embeddings_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
def creating_embeddings(text_chunks):
    return embeddings_model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=False)
embeddings = creating_embeddings(chunks)

# FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype("float32"))

#Context
def search_context(question, amount=5): 
    embedding_pergunta = embeddings_model.encode([question], convert_to_numpy=True)
    distance, indexes = index.search(embedding_pergunta.astype("float32"), amount)
    results = []
    for i in indexes[0]:
        if 0 <= i < len(chunks):
            results.append(chunks[i])
    return "\n".join(results)

def ask_AI(question):
    context = search_context(question)
    prompt = f"AI persona: {persona}, context: {context}"
    response = client.chat.completions.create(model=AI_model, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}], 
                                              temperature=Temperature, max_tokens=Max_Tokens)
    return response.choices[0].message.content

def AI_speaking(AI_answer):
    ai = pyttsx3.init()
    ai.setProperty('rate', 180)  #speed it will speak
    voices = ai.getProperty('voices')
    for voice in voices:
        voice_info = f"{voice.name} {voice.id}".lower()
        if "brazil" in voice_info or "portuguese" in voice_info:
            ai.setProperty("voice", voice.id)
            break
    ai.say(AI_answer) #what the AI will say
    ai.runAndWait()
    ai.stop()

print("Initializing audio test.")
AI_speaking("Fala isso aqui Giu.")
print("Audio test finalized.")
print("Document hash:", data_hash)
