import argparse
import json
from lib.text_processing import normalize_text, tokenize, remove_stopwords, stem_words

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            with open("data/movies.json", "r") as f:
                data = json.load(f)

            movies = data["movies"] if isinstance(data, dict) else data

            query = stem_words(remove_stopwords(tokenize(normalize_text(args.query))))
            print(f"Normalized query: {query}")

            result = []

            for movie in movies:
                title = movie.get("title", "")
                normalized_title = stem_words(remove_stopwords(tokenize(normalize_text(title))))
                if any(
                    query_word in title_word
                    for query_word in query
                    for title_word in normalized_title
                ):
                    result.append(movie)

            print(f"Searching for: {args.query}")

            for count, movie in enumerate(result[:5], start=1):
                print(f"{count}. {movie['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()