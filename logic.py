from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings

def logicOfLLM():
    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key="AIzaSyAHNURFa5FbX-stoiQ-AZ13HtftL06Mg7U",temperature=0.9,convert_system_message_to_human=True)

    # pdf_loader = PyPDFLoader("Airport_Rules_Regs_7_27_22.pdf")
    # pages = pdf_loader.load_and_split()


    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5000)
    # context = "\n\n".join(str(p.page_content) for p in pages)
    # texts = text_splitter.split_text(context)

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyAHNURFa5FbX-stoiQ-AZ13HtftL06Mg7U")
    embeddings = GPT4AllEmbeddings()

    vector_index = FAISS.load_local("faiss_index", embeddings).as_retriever(search_kwargs={"k":10})

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True
    )
    return qa_chain
