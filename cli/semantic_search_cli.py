import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the embedding model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Utility to embed text")
    embed_text_parser.add_argument("text", type=str, help="Search text")

    subparsers.add_parser(
        "verify_embeddings", help="Verify the embeddings for the movie dataset"
    )

    embed_query_parser = subparsers.add_parser("embed_query", help="Utility to embed search query")
    embed_query_parser.add_argument("query", type=str, help="Search query")

    semantic_search_parser = subparsers.add_parser("search", help="Semantic search")
    semantic_search_parser.add_argument("query", type=str, help="Search query")
    semantic_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Limit the results of the semantic search")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()