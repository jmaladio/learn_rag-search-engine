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
        if os.path.exists("cache/movies_embeddings.npy"):
            self.embeddings = np.load("cache/movies_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)

        similarity_list = list()

        for index, doc_embed in enumerate(self.embeddings):
            similarity_list.append(
                (cosine_similarity(query_embedding, doc_embed), index)
            )

        sorted_similarity_list = sorted(similarity_list, key=lambda x: x[0], reverse=True)

        list_to_dict = []
        for score, index in sorted_similarity_list[:limit]:
            doc = self.documents[index]
            list_to_dict.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })

        return list_to_dict


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

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query, limit):
    semantic_search_instance = SemanticSearch()
    with open("data/movies.json", "r") as f:
        documents = json.load(f)

    semantic_search_instance.load_or_create_embeddings(documents["movies"])
    results = semantic_search_instance.search(query, limit)

    for i, result in enumerate(results):
        print(
            f"{i + 1}. {result['title']} (score: {result['score']})\n"
            f"{result['description']}"
        )
