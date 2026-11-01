import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the embedding model")

    embed_parser = subparsers.add_parser("embed_text", help="Utility to embed text")
    embed_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser(
        "verify_embeddings", help="Verify the embeddings for the movie dataset"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()