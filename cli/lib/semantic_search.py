from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if len(text.strip()) == 0:
            raise ValueError("Text is empty")

        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        movie_list = []
        for document in documents:
            self.document_map[document["id"]] = document
            movie_list.append(f"{document["title"]}: {document["description"]}")

        self.embeddings = self.model.encode(movie_list, show_progress_bar=True)
        np.save('cache/movies_embeddings.npy', self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        movie_list = []
        for document in documents:
            self.document_map[document["id"]] = document
            movie_list.append(f"{document["title"]}: {document["description"]}")
        if os.path.exists('cache/movie_embeddings.npy'):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

        

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()

    with open("data/movies.json", "r") as f:
        documents = json.load(f)

    embeddings = semantic_search.load_or_create_embeddings(documents["movies"])

    print(f"Number of docs:   {len(documents['movies'])}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )