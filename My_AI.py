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
from pypdf import PdfReader

# API KEY
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY couldn't be found at .env file.")
client = Groq(api_key=api_key)

AI_model = "openai/gpt-oss-120b"
Temperature = 0.5
Max_Tokens = 500


# Storage paths
storage = Path(__file__).resolve().parent / "rag_storage"
storage.mkdir(exist_ok=True)

chunks_path = storage / "chunks.pkl"
embeddings_path = storage / "embeddings.npy"
index_path = storage / "index.faiss"
metadata_path = storage / "metadata.json"


# AI Persona
persona = ""
with open("AI_Persona.txt", "r", encoding="utf-8") as content:
    persona = content.read()


#Pdf
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"
    
    return text


# File selection
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select the file you want to upload.",
    filetypes=[("Text files", "*.txt"), ("Pdf files", "*.pdf")]
)
root.destroy()

if not file_path:
    print("No file selected or found.")
    exit()

if file_path.lower().endswith(".txt"):
    with open(file_path, "r", encoding="utf-8") as data:
        data_base = data.read()
elif file_path.lower().endswith(".pdf"):
    data_base = read_pdf(file_path)
else:
    raise ValueError("File type is not supported by this program")

print("Database successfully loaded!")


# Document hash
data_hash = hashlib.sha256(
    data_base.encode("utf-8")
).hexdigest()


# Embeddings model name
embeddings_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Checking storage for information
def check_storage():
    files_exist = (
        chunks_path.exists() and
        embeddings_path.exists() and
        index_path.exists() and
        metadata_path.exists()
    )

    if not files_exist:
        return False

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata["data_hash"] != data_hash:
        return False

    if metadata["embeddings_model"] != embeddings_model_name:
        return False

    return True


# Saving the knowledge
def save_knowledge(chunks, embeddings, index):

    # Saving chunks
    with open(chunks_path, "wb") as file:
        pickle.dump(chunks, file)

    # Saving embeddings
    np.save(embeddings_path, embeddings)

    # Saving FAISS index
    faiss.write_index(index, str(index_path))

    # Saving metadata
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


# Embeddings
def creating_embeddings(text_chunks):
    return embeddings_model.encode(
        text_chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )


# Embeddings model
embeddings_model = SentenceTransformer(embeddings_model_name)


# Checking storage
if check_storage():
    print("Existing knowledge found! Loading...")
    # Loading chunks
    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)
    # Loading embeddings
    embeddings = np.load(embeddings_path)
    # Loading FAISS index
    index = faiss.read_index(str(index_path))

    print("Knowledge successfully loaded!")
else:
    print("No compatible storage found. Creating knowledge...")
    # Chunks
    chunks = creating_chunks(data_base)
    # Embeddings
    embeddings = creating_embeddings(chunks)
    # FAISS
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    save_knowledge(chunks, embeddings, index)


# Searching context
def search_context(question, amount=5):
    embedding_pergunta = embeddings_model.encode(
        [question],
        convert_to_numpy=True
    )
    distance, indexes = index.search(
        embedding_pergunta.astype("float32"),
        amount
    )

    results = []
    for i in indexes[0]:
        if 0 <= i < len(chunks):
            results.append(chunks[i])
    return "\n".join(results)


# Asking AI
def ask_AI(question):
    context = search_context(question)
    prompt = f"AI persona: {persona}, context: {context}"

    response = client.chat.completions.create(
        model=AI_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=Temperature,
        max_tokens=Max_Tokens
    )
    return response.choices[0].message.content


# AI speaking
def AI_speaking(AI_answer):
    ai = pyttsx3.init()
    ai.setProperty("rate", 140) #speed of the speech
    voices = ai.getProperty("voices")

    for voice in voices:
        voice_info = f"{voice.name} {voice.id}".lower()
        #if "brazil" in voice_info or "portuguese" in voice_info:
            #ai.setProperty("voice", voice.id)
            #break
        if "united states of america" in voice_info or "english" in voice_info:
            ai.setProperty("voice", voice.id)
    ai.say(AI_answer)
    ai.runAndWait()
    ai.stop()


# Audio test
print("Initializing audio test.")
AI_speaking("Talk to me human")
print("Audio test finalized.")
print("Document hash:", data_hash)
