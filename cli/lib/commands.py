from lib.inverted_index import InvertedIndex
from lib.text_processing import tokenize_and_normalize

def bm25_idf_command(term: str) -> float:
    """
    Get the term score in the collection using the BM25 ranking system

    Args:
        term (str): The token to be searched in the collection
    Returns:
        float: The score of the term
    """
    try:
        inverted_index = InvertedIndex()
        inverted_index.load()
        tokenized_term = tokenize_and_normalize(term)
        return inverted_index.get_bm25_idf(tokenized_term)
    except Exception as e:
        print(f"Error calculating IDF: {e}")
        return 0.0

def bm25_tf_command(doc_id:int, term:str, k1:float, b:float) -> float:
    try:
        inverted_index = InvertedIndex()
        inverted_index.load()
        tokenized_term = tokenize_and_normalize(term)
        return inverted_index.get_bm25_tf(doc_id, tokenized_term, k1, b)
    except Exception as e:
        print(f"Error calculating BM25 TF: {e}")
        return 0.0