from lib.text_processing import normalize_text, tokenize_text, remove_stopwords, stem_words
from lib.data_loader import load_movies
import pickle
import os

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        # Maps IDs to Counter objects
        self.term_frequencies = {}

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

    def add_documents(self, term: str) -> list[int]:
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
        Saves the index and docmap attributes to disk using the pickle module's dump function.

        Returns:
            None
        """

        os.makedirs("cache", exist_ok=True)
        pickle.dump(self.index, open("cache/index.pkl", "wb"))
        pickle.dump(self.docmap, open("cache/docmap.pkl", "wb"))
        pickle.dump(self.term_frequencies, open("cache/term_frequencies.pkl", "wb"))

    def load(self) -> None:
        """
        Loads the index and docmap attributes from disk using the pickle module's load function.

        Returns:
            None
        """
        try:
            self.index = pickle.load(open("cache/index.pkl", "rb"))
            self.docmap = pickle.load(open("cache/docmap.pkl", "rb"))
            self.term_frequencies = pickle.load(open("cache/term_frequencies.pkl", "rb"))
        except FileNotFoundError:
            print("Index or docmap file not found. Please build the index first.")
            self.index = {}
            self.docmap = {}
            self.term_frequencies = {}

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

    