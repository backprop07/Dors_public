def embed(file, directory_prefix, chunk_size, chunk_overlap):
    loader = ''
    type = file.split('.')[-1]
    if type == 'txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    elif type == 'pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif type == 'csv':
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file)
    elif type == 'docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif type == 'html':
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        loader = UnstructuredHTMLLoader(file)
    else:
        print('Document type not supported')
    try:
        doc = loader.load()
    except Exception as e:
        print('Document type is not supported', e)
        return -1
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import GPT4AllEmbeddings
    from langchain.storage import LocalFileStore
    import os
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    string = ""
    for text in doc:
        string += text.page_content
    texts = text_splitter.create_documents([string])
    underlying_embeddings = GPT4AllEmbeddings()
    fs = LocalFileStore(f"./{directory_prefix}_cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, fs, namespace="GPT4ALL")
    db = FAISS.from_documents(texts, cached_embedder)
    if os.path.exists(f"./{directory_prefix}_faiss_index"):
        db_o = FAISS.load_local(f"{directory_prefix}_faiss_index", embeddings=underlying_embeddings,allow_dangerous_deserialization=True)
        db_o.merge_from(db)
        db = db_o
    db.save_local(f"{directory_prefix}_faiss_index")

def output_text(file):
    loader = ''
    type = file.split('.')[-1]
    if type == 'txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    elif type == 'pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif type == 'csv':
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file)
    elif type == 'docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif type == 'html':
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        loader = UnstructuredHTMLLoader(file)
    else:
        print('Document type not supported')
    try:
        doc = loader.load()
    except Exception as e:
        print('Document type is not supported', e)
        return -1
    string = ""
    for text in doc:
        string += text.page_content
    return text

def embed_semantic(file):
    embed(file, 'semantic', 700, 70)


def embed_episodic(file):
    embed(file, 'episodic', 1000, 100)


def initialize():
    import os
    for _, _, files in os.walk('./semantic_text'):
        for file in files:
            embed_semantic('./semantic_text/' + file)
    embed_episodic('./episodic_text/original.docx')


if __name__ == "__main__":
    import os

    for _, _, files in os.walk('./semantic_text'):
        for file in files:
            embed_semantic('./semantic_text/' + file)
    embed_semantic('./episodic_text/original.docx')
