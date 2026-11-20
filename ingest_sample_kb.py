from pathlib import Path
from app.core.config import get_settings
from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents

settings = get_settings()

folder = Path(settings.sample_kb_dir)
files = [p for p in folder.iterdir() if p.is_file()]

all_docs = []

for path in files:
    all_docs.extend(load_file(path))
chunks = chunk_documents(all_docs)
ids = add_documents(chunks)

print("Total chunks:", len(chunks))
print("Total indexed vectors:", len(ids))

print("\nPinecone Vector IDs:")
for vector_id in ids:
    print(vector_id)


print(f"Indexed {len(files)} files -> {len(chunks)} chunks -> {len(ids)} Pinecone vectors")


# vector-id
# 2fd8ec8c-01e4-453e-a815-1d5879cf9647
# 2bd9c654-6da6-4675-9229-124ba2c2ba57
# 2e3fa493-d46e-4f90-89f9-c8bb877fb0d5
# 5e119c24-f34d-4ce4-8a80-b00ae51a8d8b
# 708c5e68-8b9b-4719-ba98-5b39763cef05