import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def main():
    
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_PROJECT"] = "langchain-study"

    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0) # モデルの定義
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small") # 埋め込みモデルの定義

    loader = GitLoader( # リポジトリを./rag/clone-git-repoにclone
        clone_url="https://github.com/mochizuki875/bird-controller.git",
        repo_path="./rag/clone-git-repo",
        branch="main",
    )

    raw_docs = loader.load() # リポジトリをロード

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # splitterの定義
    splited_docs = text_splitter.split_documents(raw_docs)  # ドキュメントをchunkに分割


    vector_store = Chroma(  # VectorStoreを定義
        collection_name="git-repository",
        embedding_function=embeddings_model,
        persist_directory="./rag/chroma_db"
    )

    # vector_store.add_documents(raw_docs)  # ドキュメントをVectorStoreに登録(分割なし)
    vector_store.add_documents(splited_docs)  # ドキュメントをVectorStoreに登録(分割あり)
    
    retriever = vector_store.as_retriever()  # VectorStoreからRetrieverを取得

    prompt = ChatPromptTemplate.from_template(
        """You are helpful assistant.以下のcontextを踏まえてユーザーからの質問に回答してください。 
        Context: {context} 
        質問: {question}"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()} # 入力がretriver(retriver.invoke())とprompt(prompt.invoke())の両方に渡される
        | prompt
        | model
        | StrOutputParser()
    )

    result = chain.invoke("Birdリソースにはどのような値を定義すれば良いですか？")
    print(result)

if __name__ == "__main__":
    main()
