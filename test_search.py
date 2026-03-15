from src.rag.vector_store import search

try:
    results = search('how to throw a jab', sport='boxing', top_k=2)
    for r in results:
        print(f"--- Score: {r['score']}")
        print(f"Text: {r['text'][:150]}...")
except Exception as e:
    print(f"Search failed: {e}")
