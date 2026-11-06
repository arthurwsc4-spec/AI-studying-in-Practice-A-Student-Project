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
from docx import Document
import sounddevice as sd
import speech_recognition as sr
from datetime import datetime

# API KEY
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY couldn't be found at .env file.")
client = Groq(api_key=api_key)

AI_model = "openai/gpt-oss-120b"
Temperature = 0.5
Max_Tokens = 300


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


# PDF
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# Word - DOCX
def read_docx(file_path):
    document = Document(file_path)
    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


# File selection
def select_document():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select a document",
        filetypes=[
            ("Supported files", "*.txt *.pdf *.docx"),
            ("Text files", "*.txt"),
            ("PDF files", "*.pdf"),
            ("Word files", "*.docx")
        ]
    )
    root.destroy()
    return file_path


# Reading documents
def read_document(file_path):
    if file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    elif file_path.lower().endswith(".pdf"):
        return read_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return read_docx(file_path)
    else:
        raise ValueError("File type is not supported.")


# Embeddings model name
embeddings_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Saving the knowledge
def save_knowledge(chunks, embeddings, index, documents):
    # Saving chunks
    with open(chunks_path, "wb") as file:
        pickle.dump(chunks, file)
    # Saving embeddings
    np.save(embeddings_path, embeddings)
    # Saving FAISS index
    faiss.write_index(index, str(index_path))
    # Saving metadata
    metadata = {
        "embeddings_model": embeddings_model_name,
        "documents": documents
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("Knowledge successfully saved!")


def load_knowledge():
    global chunks, embeddings, index, documents
    files_exists = (
        chunks_path.exists() and
        embeddings_path.exists() and
        index_path.exists() and
        metadata_path.exists()
    )

    if files_exists:
        # Loading metadata
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        # Checking embeddings model
        if metadata["embeddings_model"] != embeddings_model_name:
            raise ValueError("Different embeddings model detected.")

        # Loading chunks
        with open(chunks_path, "rb") as file:
            chunks = pickle.load(file)

        # Loading embeddings
        embeddings = np.load(embeddings_path)

        # Loading FAISS index
        index = faiss.read_index(str(index_path))

        # Loading documents
        if "documents" in metadata:
            documents = metadata["documents"]

        elif "data_hash" in metadata:
            documents = [{
                "hash": metadata["data_hash"],
                "name": "Previous document"
            }]
        else:
            raise ValueError("Invalid storage metadata.")

        # Checking consistency
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and chunks are inconsistent.")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunks are inconsistent.")
        if embeddings.shape[1] != index.d:
            raise ValueError("Embedding dimensions are inconsistent.")
        print("Knowledge successfully loaded!")

    else:

        # Checking incomplete storage
        if any(path.exists() for path in [chunks_path, embeddings_path, index_path, metadata_path]):
            raise ValueError("Storage is incomplete. Check rag_storage.")
        print("No existing knowledge found. Creating empty storage...")

        chunks = []
        dimension = embeddings_model.get_sentence_embedding_dimension()
        embeddings = np.empty((0, dimension), dtype="float32")
        index = faiss.IndexFlatL2(dimension)
        documents = []


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

# Loading storage
load_knowledge()


# Searching context
def search_context(question, amount=5):
    if index.ntotal == 0:
        return ""
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


def add_new_document():  #needs to finish
    global chunks, embeddings, index, documents
    file_path = select_document()
    if not file_path:
        print("No document selected")
        return

    data = read_document(file_path)

    if not data.strip():
        print("Document is empty")

    document_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
    for document in documents:
        if document["hash"] == document_hash:
            print("This document already exists")
            return

    new_chunks = creating_chunks(data)
    if not new_chunks:
        print("No chunks were created")
        return

    new_embeddings = creating_embeddings(new_chunks).astype("float32")
    if new_embeddings.shape[1] != index.d:
        raise ValueError("Embeddings dimension don't match")

    index.add(new_embeddings)

    embeddings = np.vstack([embeddings, new_embeddings])

    chunks.extend(new_chunks)

    documents.append({
        "hash": document_hash,
        "name": Path(file_path).name
    })

    save_knowledge(chunks, embeddings, index, documents)

    AI_speaking("New document succesfully uploaded")


# AI Speech Recognition
def listening():
    recognizer = sr.Recognizer()
    sample_rate = int(sd.query_devices(kind="input")["default_samplerate"])

    silence_limit = 1.5
    threshold = 500
    block_duration = 0.1
    silence_time = 0
    speech_detected = False
    audio_frames = []

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        while True:
            audio_block, overflowed = stream.read(
                int(sample_rate * block_duration)
            )
            volume = np.abs(audio_block).mean()
            if volume > threshold:
                speech_detected = True
                silence_time = 0
                audio_frames.append(audio_block.copy())
            elif speech_detected:
                audio_frames.append(audio_block.copy())
                silence_time += block_duration
                if silence_time >= silence_limit:
                    break

    audio_data = np.concatenate(audio_frames)
    audio = sr.AudioData(
        audio_data.tobytes(),
        sample_rate,
        2
    )

    try:
        text = recognizer.recognize_google(
            audio,
            language="pt-BR"
        )
        return text
    except sr.UnknownValueError:
        print("Não consegui entender o áudio.")
    except sr.RequestError as error:
        print("Erro no serviço de reconhecimento:", error)
    return None


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
    ai.setProperty("rate", 240)  # speed of the speech
    voices = ai.getProperty("voices")

    for voice in voices:
        voice_info = f"{voice.name} {voice.id}".lower()
        if "brazil" in voice_info or "portuguese" in voice_info:
            ai.setProperty("voice", voice.id)
            break
        if "united states of america" in voice_info or "english" in voice_info:
            ai.setProperty("voice", voice.id)

    ai.say(AI_answer)
    ai.runAndWait()
    ai.stop()


def main():
    chat_log = []

    AI_speaking("""Olá! Sou o assistente virtual especializado no curso de Inteligência Artificial do Ensino Médio.
                    Eu fui criada por um aluno do 2 ano do curso de IA que se chama Arthur Wendler. 
                    Eu posso apresentar o curso, explicar os conteúdos estudados, esclarecer dúvidas sobre tecnologias como 
                    Machine Learning, LLMs, Embeddings e RAG, além de ajudar você a compreender como a Inteligência 
                    Artificial funciona. O que gostaria de saber?""") #introduction
    
    while True:
        question = listening()
        if question is None:
            continue
        command = question.lower().strip().rstrip(".!?")
        if command in ["encerrar conversa", "finalizar processo", "sair", "parar"]:
            print("Finalizing conversation...")
            break
        if command in ["adicionar arquivos", "adicionar documentos"]:
            try:
                add_new_document()
            except Exception as e:
                print(f"Some error occured when trying to upload the new document. Error: {e}")

        try:
            print(question)
            answer = ask_AI(question)

            print("\n\nAI:", answer)

            AI_speaking(answer)

            chat_log.append(f"Você: {question}")
            chat_log.append(f"IA: {answer}")
            chat_log.append("")
        except Exception as e:
            print(e, "An error has occurred.")

    context_folder = Path("Conversations")
    context_folder.mkdir(exist_ok=True)

    file_name = datetime.now().strftime("conversation_%Y-%m-%d_%H-%M-%S.txt")
    file_path = context_folder / file_name

    with open(file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(chat_log))

    print(f"\nConversation saved at: {file_path}")


if __name__ == "__main__":
    main()
