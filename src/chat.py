"""Interactive CLI."""
import argparse
from .retrieval import Retriever
from .answer import answer_question

def main():
    p = argparse.ArgumentParser(description="Ask questions about the 2026 Medicare NCCI Policy Manual.")
    p.add_argument("--strategy", choices=["fixed", "structure"], default="structure",
                   help="Which indexed chunk collection to query")
    p.add_argument("--retrieval", choices=["vector", "bm25", "hybrid"], default="hybrid")
    p.add_argument("-k", type=int, default=6, help="Number of context chunks")
    args = p.parse_args()
    retriever = Retriever(args.strategy)
    print("\\n2026 Medicare NCCI Manual Q&A")
    print(f"Chunking={args.strategy} | Retrieval={args.retrieval} | top_k={args.k}")
    print("Ask a question. Type 'exit' to quit.\\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\nGoodbye.")
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        hits = retriever.search(question, args.retrieval, args.k)
        if not hits:
            print("\\nAssistant> I could not retrieve relevant passages.\\n")
            continue
        print("\\nAssistant> " + answer_question(question, hits))
        print("\\nSources retrieved:")
        for h in hits:
            print(f"  - PDF pp. {h['page_start']}-{h['page_end']} | {h.get('section','')}")
        print()

if __name__ == "__main__":
    main()
