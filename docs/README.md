# RAG Search System Documentation

This documentation follows the Boot.dev course sequence for building a retrieval-augmented generation (RAG) search system.

The chapters combine three perspectives:

1. Course concepts and terminology.
2. The implementation currently present in this repository.
3. Broader information-retrieval and AI-engineering context.

Full chapter content will be written as the corresponding course material is completed and provided. Until then, each chapter file records its place in the learning path and the questions it will answer.

## Chapter Roadmap

| Chapter | Topic | Status |
| --- | --- | --- |
| [01](01-preprocessing.md) | Preprocessing | Complete |
| [02](02-tf-idf.md) | TF-IDF | Complete |
| [03](03-keyword-search.md) | Keyword Search | Upcoming |
| [04](04-semantic-search.md) | Semantic Search | Upcoming |
| [05](05-chunking.md) | Chunking | Upcoming |
| [06](06-hybrid-search.md) | Hybrid Search | Upcoming |
| [07](07-llms.md) | LLMs | Upcoming |
| [08](08-reranking.md) | Reranking | Upcoming |
| [09](09-evaluation.md) | Evaluation | Upcoming |
| [10](10-augmented-generation.md) | Augmented Generation | Upcoming |
| [11](11-agentic.md) | Agentic Systems | Upcoming |
| [12](12-multimodal.md) | Multimodal Systems | Upcoming |

## How Chapters Are Written

Each chapter uses [the chapter template](TEMPLATE.md) as a starting point. A finished chapter should explain the idea, connect it to this repository, show a small working example, identify common mistakes, and provide references for deeper study.

The repository implementation is authoritative for statements about this project. General best practices and production alternatives are labeled separately rather than presented as if they were already implemented here.

## Project Verification

Run commands from the repository root. The current CLI workflow is:

```bash
python cli/keyword_search_cli.py build
python cli/keyword_search_cli.py search "space adventure"
python cli/keyword_search_cli.py tf 1 police
python cli/keyword_search_cli.py idf police
python cli/keyword_search_cli.py tfidf 1 police
```

The index must be built before lookup commands can use the cache.
