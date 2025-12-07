import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# السطر 6: تأكد أنها FastEmbedEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="SLAA AI Sponsor", page_icon="🛡️")
st.title("🛡️ رفيق التعافي (نسخة الويب السريعة)")

if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("مفتاح API غير موجود! تأكد من إضافته في Secrets.")
    st.stop()

@st.cache_resource
def load_library():
    folder_path = "library" 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return None
    if not os.listdir(folder_path):
        return "EMPTY"

    with st.spinner("جاري بناء قاعدة المعرفة..."):
        loader = PyPDFDirectoryLoader(folder_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        # السطر 40: المحرك الجديد السريع
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore

vectorstore = load_library()

if not vectorstore or vectorstore == "EMPTY":
    st.warning("⚠️ المكتبة فارغة. تأكد من وجود ملفات PDF داخل مجلد 'library'.")
    st.stop()

system_prompt = (
    "You are an expert SLAA Sponsor using the Llama-3-70b model. "
    "Rules: 1. Answer in Arabic only. 2. Be strict but kind. 3. Use context."
    "\nContext: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
retriever = vectorstore.as_retriever()
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("تحدث مع موجهك..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("جاري الاتصال..."):
            response = chain.invoke({"input": user_input})
            st.markdown(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
