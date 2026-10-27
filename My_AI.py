import os
from getpass import getpass
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import pyttsx3
from tkinter import Tk, filedialog
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY couldn't be found at .env file.")
client = Groq(api_key=api_key)
AI_model = "openai/gpt-oss-120b"
Temperature = 0.5
Max_Tokens = 500

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
AI_speaking("My name is Arthur and I'm the one who made this project.")
print("Audio test finalized.")
