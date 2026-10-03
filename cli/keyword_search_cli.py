import argparse
from lib.data_loader import load_movies
from lib.text_processing import normalize_text, tokenize_text, remove_stopwords, stem_words
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()