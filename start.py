def setup():
    print("Warning, if you are under the age of 18, please use this app accompanied by legal guardian.")
    print("Hello, I am Dors.")
    user_input=input("May I know your name please?\n")
    from langchain.llms import GPT4All
    from embed import initialize,output_text
    initialize()
    llm = GPT4All(
        model=r"E:\GPT4ALL\model\wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin",
        max_tokens=2048,
    )
    name=llm.predict(f"Example:\n User's input: My name is John.\nUser's name: John \n\nUser's input: {user_input}. "
                     f"\nUser's name:")
    system_memory=output_text('./episodic_text/original.docx')
    system_memory=llm.predict(f'Given:{system_memory}\nInfer and summarize the information and characteristic of #person0#:')
    import pickle
    import dbm
    with dbm.open("user",'c') as user:
        user["name"]=pickle.dumps(name)
        user["system_memory"]=pickle.dumps(system_memory)
    print(f"Nice to meet you, {name}.\nHow can I help you?")
    memory=f"#person0#:Hello, I am Dors.May I know your name please?\n #person1#:{user_input}\n"
    "#person0#:Nice to meet you, {name}.\nHow can I help you?"
    return memory
