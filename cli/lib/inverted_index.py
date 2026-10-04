from lib.text_processing import normalize_text, tokenize_text, remove_stopwords, stem_words
from lib.data_loader import load_movies
from lib.constants import BM25_K1, BM25_B
import pickle
import os
import math

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        # Maps IDs to Counter objects
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Normalize the input text and then use tokenize_text helper function  on the input text and
        add each token and document ID to the index.

        Args:
            doc_id (int): The unique identifier for the document.
            text (str): The text content of the document.
        Returns:
            None
        """
        tokens = stem_words(remove_stopwords(tokenize_text(normalize_text(text))))
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
                self.term_frequencies[token] = {}
            self.index[token].add(doc_id)
            self.term_frequencies[token][doc_id] = self.term_frequencies[token].get(doc_id, 0) + 1

        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        """
        Calculate the average document length in the collection.

        Returns:
            float: The average document length.
        """
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        """
        Get the document IDs associated to the given term
        and return them as a sorted list in ascending order

        Args:
            term (str): The token to be searched in documents
        Returns:
            list[int]: A list of IDs sorted in ascending order
        """

        if term in self.index:
            return sorted(self.index[term])
        else:
            return []

    def build(self) -> None:
        """
        Build the inverted index from the provided documents.
        This method should be called after all documents have been added.

        Returns:
            None
        """

        movies = load_movies("data/movies.json")
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        """
        Saves the index, docmap, term_frequencies, and doc_lengths attributes to disk using the pickle module's dump function.

        Returns:
            None
        """

        os.makedirs("cache", exist_ok=True)
        pickle.dump(self.index, open("cache/index.pkl", "wb"))
        pickle.dump(self.docmap, open("cache/docmap.pkl", "wb"))
        pickle.dump(self.term_frequencies, open("cache/term_frequencies.pkl", "wb"))
        pickle.dump(self.doc_lengths, open(self.doc_lengths_path, "wb"))

    def load(self) -> None:
        """
        Loads the index, docmap, term_frequencies, and doc_lengths attributes from disk using the pickle module's load function.

        Returns:
            None
        """
        try:
            self.index = pickle.load(open("cache/index.pkl", "rb"))
            self.docmap = pickle.load(open("cache/docmap.pkl", "rb"))
            self.term_frequencies = pickle.load(open("cache/term_frequencies.pkl", "rb"))
            self.doc_lengths = pickle.load(open(self.doc_lengths_path, "rb"))
        except FileNotFoundError:
            print("Index or docmap file not found. Please build the index first.")
            self.index = {}
            self.docmap = {}
            self.term_frequencies = {}
            self.doc_lengths = {}

    def get_tf(self, doc_id:int, term:str) -> int:
        """
        Get the term frequency of a term in a specific document.

        Args:
            doc_id (int): The unique identifier for the document.
            term (str): The token to be searched in the document.
        Returns:
            int: The term frequency of the term in the specified document.
        """
        return self.term_frequencies.get(term, {}).get(doc_id, 0)

    def get_bm25_idf(self, term:str) -> float:
        """
        Get the term score in the collection using the BM25 ranking system

        Args:
            term (str): The token to be searched in the collection
        Returns:
            float: The score of the term
        """
        n = self.docmap.__len__()
        df = self.get_documents(term).__len__()
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id:int, term:str, k1:float = BM25_K1, b:float = BM25_B) -> float:
        raw_tf = self.get_tf(doc_id, term)

        # Length normalization factor
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)

        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm) if raw_tf > 0 else 0.0

    def bm25(self, doc_id: int, term: str) -> float:
        """
        Calculate the BM25 score for a given document ID and term.

        Args:
            doc_id (int): The unique identifier for the document.
            term (str): The token to be searched in the document.
        Returns:
            float: The BM25 score for the term in the specified document.
        """
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_idf * bm25_tf

    def bm25_search(self, query, limit):
        """
        Search for documents using the BM25 ranking system.

        Args:
            query (str): The search query.
            limit (int): The maximum number of results to return.

        Returns:
            list[tuple[int, float]]: Document IDs and scores sorted by BM25 score.
        """
        query_terms = stem_words(remove_stopwords(tokenize_text(normalize_text(query))))
        scores = {}

        for term in query_terms:
            for doc_id in self.get_documents(term):
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += round(self.bm25(doc_id, term), 2)

        # Sort documents by score in descending order and return the top 'limit' results
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:limit]

