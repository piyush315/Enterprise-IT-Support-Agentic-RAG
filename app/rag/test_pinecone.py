
from app.rag.vectorstore import get_vectorstore

store = get_vectorstore()

query = """
According to the Reliance Annual Report for the year
ended 31st March 2026, what was Reliance Industries'
consolidated revenue from operations?
"""

results = store.similarity_search_with_score(
    query,
    k=5
)

print("Total results:", len(results))

for i, (doc, score) in enumerate(results, 1):

    print(f"\n========== RESULT {i} ==========")

    print("Score:", score)

    print("Source:", doc.metadata.get("source"))

    print("Page:", doc.metadata.get("page"))

    print("Content:", doc.page_content[:500])
