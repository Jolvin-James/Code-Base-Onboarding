from ingestion.loader import load_documents

if __name__ == "__main__":
    documents = load_documents("docs")

    print("\nSample Document Structure:\n")
    print(documents[0])