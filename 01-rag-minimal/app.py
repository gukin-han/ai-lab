# app.py
from rag_pipeline import RAGPipeline

def main():
    rag = RAGPipeline(docs_path="docs", top_k=3)
    rag.build_index()

    print("🔎 RAG 데모입니다. 'exit' 치면 종료.\n")
    while True:
        q = input("Q> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        answer = rag.query(q)
        print("\nA>\n", answer, "\n")

if __name__ == "__main__":
    main()
