#importing streamlit for frontend
import streamlit as st

#importing os for path 
import os

#importing tempfile for storing documents temporarily in the staging area
import tempfile

from pathlib import Path

#import langchain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

#importing langchain loaders for handling various file types
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, JSONLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import pickle
import json
from langchain_core.documents import Document

# directory for storing vector
vector_directory = "chroma_db"

#embedding model used
#embedding_model_name = "all-MiniLM-L6-v2"
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"

#llm model to be used
llm_model_name = "gemma3:1b"

#cross encoder model for re ranking document chunks
cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#defining prompt template
Prompt_Template = ChatPromptTemplate.from_template(
    """You are a helpful document Q&A assistant. 
    Use the context provided below to answer the user's question.
    If the context contains relevant information, provide a detailed answer.
    If you cannot find relevant information in the context, say so politely.

    CONTEXT: {context}

    QUESTION: {question}
    
    ANSWER:"""
)

@st.cache_resource

# function to get the embedding model for embedding documents
def embedding_documents():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name = embedding_model_name,
                                                model_kwargs = {'device':'cpu'},
                                                encode_kwargs = {'normalize_embeddings':True})
        return embedding_model
    except Exception as e:
        # incase embedding model can't be called
        st.error(f"Error Occured! Can't initialize embedding model : {e}")
        return None
    
@st.cache_resource

#function to load cross encoder for re ranking
def load_cross_encoder():
    try:
        cross_encoder = CrossEncoder(cross_encoder_model_name)
        return cross_encoder
    except Exception as e:
        st.error(f"Error! Can't load Cross-Encoder : {e}")
        return None
        
#function to chunk and load the documents 
def chunking_and_loading(directory):
    documents = []

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        # for pdf files
        if file_name.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
    
        #for docx file
        elif file_name.endswith((".doc", ".docx")):
            loader = UnstructuredWordDocumentLoader(file_path)
            
        #for text file
        elif file_name.endswith((".txt", ".md")):
            loader = TextLoader(file_path, encoding = 'utf-8')
            
        #for json file
        elif file_name.endswith(".json"):
            try:
                with open(file_path, 'r', encoding = 'utf-8') as f:
                    json_data = json.load(f)
                json_text = json.dumps(json_data, indent = 2, ensure_ascii = False)
                doc = Document(
                    page_content = json_text,
                    metadata = {"source":file_path, "file_type":"json"}
                )
                documents.append(doc)
                st.info(f"Loaded {file_name}:{len(json_text)} characters")
                continue
            except Exception as e:
                st.warning(f"Could Not Load {file_name} : {e}")
                continue
            
        #for pickle
        elif file_name.endswith(".pkl"):
            try:
                with open(file_path, 'rb') as f:
                    pkl_data = pickle.load(f)
                    
                #converting pickle data to string 
                pkl_text = str(pkl_data)
                if isinstance(pkl_data, dict):
                    pkl_text = json.dumps(pkl_data, indent = 2, default = str, ensure_ascii = False)
                elif isinstance(pkl_data, (list, tuple)):
                    pkl_text = '\n'.join([str(item) for item in pkl_data])
                doc = Document(
                    page_content = pkl_text,
                    metadata = {"source": file_path, "file_type": "pickle"}
                )
                documents.append(doc)
                st.info(f"Loaded {file_name} : {len(pkl_text)} characters")
                continue
            except Exception as e:
                st.warning(f"Could Not Load {file_name} : {e}")
                continue
        
        #for excel files
        elif file_name.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(file_path, model = "elements")
            
        #for unsupported file types
        else:
            st.warning(f"Unsupported file type : {file_name}")
            continue
        try:
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Could Not Load {file_name}: {e}. Unsupported file type")
            continue 
        
    ##Splitting the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len
    )

    return text_splitter.split_documents(documents)
      
#function to create a vector and store in chroma 
def vectorisation_and_store(uploaded_files, embedding_model, vector_directory):
    if not uploaded_files:
        return None, 0
    
    #creating a temporary directory to store uploded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, file.name)
            Path(temp_file_path).write_bytes(file.getvalue())

        #loading chunked documents
        document_chunks = chunking_and_loading(temp_dir)

        if not document_chunks:
            return None, 0

        #intializing vector storage
        vector_store = Chroma.from_documents(
            documents = document_chunks,
            embedding = embedding_model,
            persist_directory = vector_directory
        )

        return vector_store, len(document_chunks)

#making RAG Chain
def initialize_rag_chain(vector_store, cross_encoder = None):
    try:
        llm = Ollama(model = llm_model_name, temperature = 0.1)

        #Retriever to fetch top documents
        initial_k = 20 if cross_encoder else 5
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": initial_k}
        )

        #Cross-encoder re-ranking function
        def rerank_documents(docs, query):
            # if no cross encoder found return top 5 documents
            if not cross_encoder or not docs:
                return docs[:5]

            #pair of query document to be given
            pairs = [[query, doc.page_content] for doc in docs]

            #Scoring each pair 
            scores = cross_encoder.predict(pairs)

            #top score documents are used
            doc_score_pairs = list(zip(docs, scores))
            doc_score_pairs.sort(key=lambda x : x[1], reverse = True)  

            #Returning top 5 re ranked documents
            reranked_docs = [doc for doc, score in doc_score_pairs[:5]]

            return reranked_docs          
        
        #joining texts from multiple inputs
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            RunnableParallel(
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
            )
            | Prompt_Template
            | llm
            | StrOutputParser()
        )

        
        #Retriever chain (Retrieval + Generation)
        def rag_chain_wrapper(query_dict):
            query = query_dict["input"]
            
            #getting initial documents using retriever
            initial_docs = retriever.invoke(query)
            
            #reranking using crossencoder
            reranked_docs = rerank_documents(initial_docs, query)
            
            #reranked documents
            context = format_docs(reranked_docs)
            answer = (Prompt_Template| llm | StrOutputParser()).invoke({
                "context":context, 
                "question":query
            })
            
            #docs = retriever.invoke(query_dict["input"])
            return {"answer": answer, "context": reranked_docs}
            
        return rag_chain_wrapper

    except Exception as e:
        st.error(f"Error Occured! Can't initialize RAG Chain ! : {e}")
        return None

def main():
    st.set_page_config(page_title = "Retrieval-Augmented Generation (RAG) Document Q&A", layout = "wide")
    st.title("Retrieval-Augmented Generation (RAG) Document Q&A")

    #creating a session 
    if 'documents_processed' not in st.session_state:
        st.session_state['documents_processed'] = False

    #loading Embedding model
    embedding_model = embedding_documents()
    
    #loading crossencoder for reranking
    cross_encoder = load_cross_encoder()

    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader( "Drag & Drop or Browse to Upload Documents (.pdf, .docx, .txt, .md, .json, .pkl, .xlsx, .xls)",
            type = ['pdf','docx','txt','md', 'json', 'pkl', 'xlsx', 'xls'],
            accept_multiple_files = True,
            help="Supported: PDF, Word, Text, Markdown, JSON, Excel, Pickle"
        )
    
        #creating reranking checkbox
        st.divider()
        
        use_reranking = st.checkbox("Re-Rank using Cross-encoder", value = True,
                                    help = "Re-rank results for better results. Might be slower ")
        if use_reranking:
            st.info("Retrieving and re-ranking top documents")
        else:
            st.info("Direct retrieval of top documents. No re-ranking")
            
        st.divider()

        # creating a button to start the indexing process
        if st.button("Process Documents", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one document first!")
            elif not embedding_model:
                st.error("Embedding model not loaded!")
            else:
                with st.spinner("Processing documents. Please Wait"):
                    vector_store, chunk_count = vectorisation_and_store(
                        uploaded_files,
                        embedding_model,
                        vector_directory
                    )

                    if vector_store and chunk_count > 0:
                        st.session_state['documents_processed'] = True
                        #st.success(f"{chunk_count} chunks are created and indexed. Ready for query.")
                    else:
                        st.error("Indexing Failed!!")
                        st.session_state['documents_processed'] = False

    if st.session_state.get('documents_processed', False):
        if st.session_state['documents_processed'] and os.path.exists(vector_directory):
            st.info("Document loaded and indexed. Ready to query!!")
        else:
            st.warning("No document indexed to query!. Please upload and press Process Document Button")
    
    if st.session_state.get('documents_processed', False):
        if os.path.exists(vector_directory):
            try:
                vector_store_persisted = Chroma(
                    persist_directory = vector_directory,
                    embedding_function = embedding_model
                )
                
                # Counting documents in vector store
                try:
                    collection = vector_store_persisted._collection
                    chunk_count = collection.count()
                    st.success(f"{chunk_count} chunks indexed and ready for query!")
                except:
                    st.success("Documents loaded and indexed. Ready to query!")
                
                rag_chain = initialize_rag_chain(vector_store_persisted, cross_encoder)
            except Exception as e:
                st.error(f"Failed to load!. Check chroma_db folder. Error: {e}")
                rag_chain = None
                st.session_state['documents_processed'] = False
        else:
            rag_chain = None
            st.session_state['documents_processed'] = False
            st.warning("No documents indexed yet. Please upload documents and press 'Process Documents' button.")
    else:
        st.warning("No documents indexed yet. Please upload documents and press 'Process Documents' button.")
        rag_chain = None


    if rag_chain:
        st.subheader("Ask Questions")
        user_query = st.text_input("Enter your question here : ", key = "user_query_input")

        if user_query:
                with st.spinner("Searching Documents for answers "):

                    #Executing RAG chain using invoke
                    response = rag_chain({"input" : user_query})

                    #answer is given by LLM
                    final_answer = response.get("answer", "No answer")
                    #context is retrieved documents
                    source_documents = response.get("context", [])

                    st.success("Answer Generated : ")
                    st.markdown(final_answer)

                    st.subheader("Sources used : ")

                    source_info = {}
                    for doc in source_documents:
                        source_path = doc.metadata.get('source', 'Unknown source')
                        page = doc.metadata.get('page',None)

                        if page is not None:
                            page = page+1
                        else:
                            page = 'N/A'

                        if source_path not in source_info:
                            source_info[source_path] = set()
                        source_info[source_path].add(page)

                    for source_path, pages in source_info.items():
                        source_name = os.path.basename(source_path)
                        sorted_pages = sorted([p for p in pages if p != "N/A"])

                        page_display = ', '.join(map(str, sorted_pages)) if sorted_pages else 'N/A'
                        st.info(f"**{source_name}** (Pages : {page_display})")

if __name__ == "__main__":
    main()