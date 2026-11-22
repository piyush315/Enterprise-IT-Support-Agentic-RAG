# # from app.core.config import get_settings

# # settings = get_settings()

# # print(f"App Name: {settings.app_name}")

# from app.services.ingestion import load_file, chunk_documents
# from pathlib import Path
# from app.rag.vectorstore import add_documents

# docs = load_file(Path("data/sample_kb/company_it_handbook.md"))
# chunks = chunk_documents(docs)

# add_documents(chunks)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)