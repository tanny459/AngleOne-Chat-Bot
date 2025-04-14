from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def clean_support_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    cleaned = []
    keep = False
    for line in lines:
        if line.startswith("### Q:"):
            cleaned.append(line.strip())
            keep = True
        elif line.startswith("**A:**") and keep:
            cleaned.append(line.strip())
            cleaned.append("")
            keep = False
    with open(output_path, 'w') as f:
        f.write("\n".join(cleaned))

def custom_split_qa(file_path):
    with open(file_path, 'r') as f:
        qa_blocks = f.read().strip().split("\n\n")
    return [Document(page_content=block) for block in qa_blocks if block.strip()]

clean_support_file("angelone_support_pages.txt", "cleaned_kb.txt")
chunks = custom_split_qa("cleaned_kb.txt")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")
