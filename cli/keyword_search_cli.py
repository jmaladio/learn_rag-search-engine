import argparse
import math
from lib.data_loader import load_movies
from lib.text_processing import normalize_text, tokenize_and_normalize, tokenize_text, remove_stopwords, stem_words
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="The document ID to search in")
    tf_parser.add_argument("term", type=str, help="The term to search for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="The term to search for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="The document ID to search in")
    tfidf_parser.add_argument("term", type=str, help="The term to search for")

    args = parser.parse_args()

    match args.command:
        case "search":
            try:
                index = InvertedIndex()
                index.load()
            except Exception as e:
                print(f"Error loading index: {e}")
                return

            query = stem_words(remove_stopwords(tokenize_text(normalize_text(args.query))))
            print(f"Normalized query: {query}")

            result = []
            for term in query:
                result.extend(index.add_documents(term))
                if len(result) == 5:
                    break

            print(f"Searching for: {args.query}")

            for movie in result:
                print(f"Movie ID: {movie}, Title: {index.docmap[movie]['title']}")
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
        case "tf":
            try:
                tokenized_term = tokenize_and_normalize(args.term)
                inverted_index = InvertedIndex()
                inverted_index.load()
                freq = inverted_index.get_tf(args.doc_id, tokenized_term)
                if freq is not None:
                    print(f"{freq}")
                else:
                    print(0)
            except Exception as e:
                print(f"Error loading index: {e}")
                return
        case "idf":
            try:
                inverted_index = InvertedIndex()
                inverted_index.load()
                tokenized_term = tokenize_and_normalize(args.term)
                idf_score = math.log((inverted_index.docmap.__len__() + 1) / (len(inverted_index.add_documents(tokenized_term)) + 1))
                print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")
            except Exception as e:
                print(f"Error calculating IDF: {e}")
                return
        case "tfidf":
            try:
                inverted_index = InvertedIndex()
                inverted_index.load()
                tokenized_term = tokenize_and_normalize(args.term)
                tf = inverted_index.get_tf(args.doc_id, tokenized_term)
                idf_score = math.log((inverted_index.docmap.__len__() + 1) / (len(inverted_index.add_documents(tokenized_term)) + 1))
                tfidf_score = tf * idf_score
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf_score:.2f}")
            except Exception as e:
                print(f"Error calculating TF-IDF: {e}")
                return
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()