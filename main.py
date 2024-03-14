if __name__ == "__main__":
    import os
    import dbm
    import random
    import pickle
    import datetime
    from langchain.llms import GPT4All
    from langchain.prompts import PromptTemplate
    from langchain.embeddings import GPT4AllEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    from langchain.retrievers.document_compressors import LLMChainExtractor
    import wikipedia

    if os.path.exists('user.dat'):
        with dbm.open('user') as user:
            name = pickle.loads(user['name']).strip()
            system_memory = pickle.loads(user['system_memory'])
        greeting = random.choice([f"It's good to see you again, {name}", f"Hi,{name}.",
                                  f"{name}, nice to meet you again.", f"Kind of miss you, {name}"])
        print(greeting)
        memory = f'Conversation starts at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} #person0#: {greeting}'
    else:
        from start import setup
        memory = setup()
        with dbm.open('user') as user:
            name = pickle.loads(user['name'])
            system_memory = pickle.loads(user['system_memory'])
    llm = GPT4All(
        model=r"E:\GPT4ALL\model\wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin",
        max_tokens=2048,
    )
    embeddings = GPT4AllEmbeddings()
    end_prompt = PromptTemplate.from_template("Determine whether user is trying to end the conversation based on its "
                                              "last input and answer Yes or No.\n\n User input:Bye. \n Answer: Yes\n "
                                              "User input: I have a great day.\n Answer: No\n\nUser"
                                              "input: It is boring.\n Answer: No\nUser input:See you later.\nAnswer: "
                                              "Yes\n\nUser input: {inp}.\n Answer:")
    episodic_to_semantic_prompt = PromptTemplate.from_template(
        "Additional information: {episodic}\n Question asked to #person0#: {question}"
        "\nExtract keywords from the additional information and question to "
        "retrieve relevant information from general"
        "knowledge database:")
    conversation_prompt = PromptTemplate.from_template(
        f"System information: You are #person0#. #person0#(Dors) and #person1#({name}) are good friend, reply to "
        f"#person1# in the"
        f"perspective of #person0#\n About #person0#:{system_memory}\n"
        'History:{episodic}\nAdditional information:{'
        'semantic}\nPotentially relevant information on the internet:{wiki}\n\nCurrent conversation:{memory}')
    inp = input()
    db1 = FAISS.load_local('./episodic_faiss_index', embeddings)
    db2 = FAISS.load_local('./semantic_faiss_index', embeddings)
    retriever1 = db1.as_retriever()
    retriever2 = db2.as_retriever()
    compressor = LLMChainExtractor.from_llm(llm)
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.3)
    compression_retriever1 = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                            base_retriever=retriever1)
    compression_retriever2 = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                            base_retriever=retriever2)
    while llm.predict(text=end_prompt.format(inp=inp)).strip() != 'Yes':
        memory += f'\n#person1#:{inp}\n\n#person0#:'
        episodic = ""
        compressed_docs = compression_retriever1.get_relevant_documents(f'#person0# is asked:{inp}')
        for doc in compressed_docs:
            episodic += doc.page_content
        episodic = episodic[:1000]
        if inp.strip() != '':
            compressed_docs1 = compression_retriever2.get_relevant_documents(inp)
            semantic_inp = llm.predict(episodic_to_semantic_prompt.format(episodic=episodic, question=inp))
        else:
            compressed_docs1 = []
            semantic_inp = ''
        if semantic_inp.strip() != '':
            compressed_docs2 = compression_retriever2.get_relevant_documents(semantic_inp)
            wiki=''
            # wikip = wikipedia.search(semantic_inp)
            # try:
            #     wiki = wikipedia.summary(wikip[0])[:1000]
            # except:
            #     print('This is an error with wikipedia, not the program.')
        else:
            compressed_docs2 = []
            wiki = ''
        semantic = ''
        for doc in compressed_docs1:
            semantic += doc.page_content
        for doc in compressed_docs2:
            semantic += doc.page_content
        semantic = semantic[:1000]
        output = llm.predict(conversation_prompt.format(episodic=episodic, semantic=semantic, wiki=wiki,memory=memory))
        output = output.split('#person1#')[0].replace('#person0#','')
        print(output)
        memory += output.replace('\n', ' ')
        inp = input()
    word_choice = ['Later', 'See you later', 'Take care',
                   'Talk to you later', 'So long', 'Bye', 'Farewell', 'Goodbye',
                   'so long', 'Catch you later', 'I\'m out', 'See you']
    if datetime.datetime.now().hour >= 23 or datetime.datetime.now().hour <= 4:
        word_choice = ['Have a nice sleep', 'Good night', 'Have a nice dream']
    output = random.choice(word_choice) + f", {name}"
    print(output)
    memory += f'\n#person1#({name}):{inp}\n#person0#:'
    memory += output.replace('\n', ' ')
    memory += f'\nConversation ends at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} '
    system_memory=llm.predict(f'Given profile of #person0#: {system_memory}\nGiven conversation:{memory}\n Infer and summarize the information '
                f'and characteristic of #person0#:')
    with dbm.open("user",'c') as user:
        user["system_memory"]=pickle.dumps(system_memory)
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain.storage import LocalFileStore
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=70,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.create_documents([memory])
    fs = LocalFileStore(f"./episodic_cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, fs, namespace="GPT4ALL")
    db = FAISS.from_documents(texts, cached_embedder)
    db1.merge_from(db)
    db1.save_local(f"episodic_faiss_index")
