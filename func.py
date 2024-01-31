import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain_community.vectorstores import FAISS,Chroma
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer, util
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.chains import LLMChain
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent,ZeroShotAgent
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever,SelfQueryRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.chains.query_constructor.base import AttributeInfo
from typing import Any
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from unstructured.partition.pdf import partition_pdf
import pytesseract
from langchain.agents import initialize_agent, AgentType
import faiss
import os
import uuid
import base64
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Set up LLM caching using an in-memory cache
# set_llm_cache(InMemoryCache())

# Title for the Streamlit app
st.title("PDF Summarizer, QA & Chat")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "chat_history" not in st.session_state :
    st.session_state.chat_history=[]
# if "tools"  not in st.session_state:
#     st.session_state.tools = None


# Check if chain exists in session state, if not, initialize it
if "chain" not in st.session_state:

    def load_openai_api_key():
        dotenv_path = "openai.env"
        load_dotenv(dotenv_path)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(f"Unable to retrieve OPENAI_API_KEY from {dotenv_path}")
        return openai_api_key
 
    # Initialize components including ChatOpenAI model and QA chain
    def initialize_components(retriever):

        OpenAIModel = "gpt-4-1106-preview"
        #memory = ConversationBufferMemory(memory_key="conversation_history")
        
        llm = ChatOpenAI(model=OpenAIModel, temperature=0, openai_api_key=load_openai_api_key())
               
        # prompt = hub.pull("rlm/rag-prompt")

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )


        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]


        rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever
            )
            | qa_prompt
            | llm
        )
        return rag_chain

    # Perform initialization and store the chain in session state
    
    # print("Initialization complete.")

# process text extracted from PDFs and create a knowledge base



def process_text(text_elements,image_elements):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500
    )
    # chunks = text_splitter.split_documents(doc)
    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # doc_id = [str(uuid.uuid4()) for _ in text_elements]
    # index_to_docstore_id = list(zip(doc_id,text_elements))

    # print(index_to_docstore_id)

    # knowledgeBase = FAISS.from_documents(chunks, embeddings)
    
    # retriever = knowledgeBase.as_retriever(search_type="mmr",
    #             search_kwargs={'k': 7, 'fetch_k': 50})
    
    store = InMemoryStore()

    vectorstore = Chroma(embedding_function = embeddings)

    retriever = ParentDocumentRetriever(
        docstore=store,
        vectorstore=vectorstore,   
        parent_splitter=text_splitter,
        child_splitter=child_text_splitter
    )
    retriever.add_documents(text_elements)
    # for i,docs in enumerate(doc):
    #     docs.metadata["id_key"]=doc_id[i]

    # vectorstore.add_documents(doc)
    # store.mset(list(zip(doc_id,image_elements)))
#     compressor = CohereRerank()
#     compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )
    
    return retriever

# relevance score between a question and response
def calculate_relevance_score(question, response):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    question_embedding = model.encode(question, convert_to_tensor=True)
    response_embedding = model.encode(response, convert_to_tensor=True)

    cosine_score = util.pytorch_cos_sim(question_embedding, response_embedding)
    relevance_score = cosine_score.item()

    return relevance_score

  
# store conversation history in session state
def store_conversation(user_query, assistant_response):
    st.session_state.conversation_history.append({"Question_number": len(st.session_state.conversation_history),"user_query": user_query, "assistant_response": assistant_response})

# display conversation history
def display_conversation_history():
    st.sidebar.subheader("Conversation History")
    for conv in st.session_state.conversation_history:
        st.sidebar.markdown(f"**User:** {conv['user_query']}")
        st.sidebar.markdown(f"**Assistant:** {conv['assistant_response']}")
        st.sidebar.markdown("---")

# Function to handle user interaction and responses
        
def handle_user_interaction(pdf_files, user_question):
    content_found = False
    relevant_responses = []
    # st.session_state.chain = initialize_components(st.session_state.tools)
    rag_chain = st.session_state.chain
    response = rag_chain.invoke({"question": user_question, "chat_history": st.session_state.chat_history})

    st.session_state.chat_history.extend([HumanMessage(content=user_question), response])
    assistant_response = response.content
    if response:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = assistant_response
            for chunk in assistant_response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        store_conversation(user_question, assistant_response)

    # if docs:
    #     content_found = True
    #     response = st.session_state.chain.run(input_documents=docs, question=user_question)

    #     if response:
    #         relevance_score = calculate_relevance_score(user_question, response)
    #         relevant_responses.append({"response": response, "score": relevance_score})

    # if not content_found:
    #     st.write("No relevant information found in the uploaded PDFs.")
    # elif relevant_responses:
    #     most_relevant = max(relevant_responses, key=lambda x: x["score"])
    #     st.session_state.messages.append({"role": "user", "content": user_question})
    #     st.session_state.messages.append({"role": "assistant", "content": most_relevant['response']})
    #     store_conversation(user_question, most_relevant['response'])
    # else:
    #     st.write("I couldn't find any relevant information about your question in the uploaded PDFs.")


# Main function for the Streamlit app
        
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def summarize_text(text_element,llm):
    prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# Function for table summaries
def summarize_table(table_element,llm):
    prompt = f"""Extract the detailed content of the table.\n{table_element}.\n\nNote: Understand the table structure and get insights from it.:\n
    
    Reminder : You shouldn't skip the 
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# Function for image summaries
def summarize_image(encoded_image,llm):
    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the contents of this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = llm.invoke(prompt)
    return response.content
def save_uploaded_file(uploaded_file):
    # Create a folder to store uploaded files if it doesn't exist
    os.makedirs("uploaded_pdfs", exist_ok=True)

    # Save the file to the folder
    file_path = os.path.join("uploaded_pdfs", uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path   
def main():
    with st.sidebar.expander("Upload your PDF Documents"):
        pdf_files = st.sidebar.file_uploader(' ', type='pdf', accept_multiple_files=True)
        input_path = os.getcwd()
        output_path = os.path.join(os.getcwd(), "output")
        if pdf_files:
            st.session_state.uploaded_files = pdf_files
            if "tools" not in st.session_state:
                doc = []
                text_elements = []
                # table_elements = []
                image_elements = []
                for pdf in pdf_files:
                    text = ""
                    raw_pdf_elements = partition_pdf(
                    filename=os.path.join(input_path, save_uploaded_file(pdf)),
                    strategy="hi_res",
                    hi_res_model_name = "detectron2_onnx",
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    max_characters=4000,
                    new_after_n_chars=3800,
                    combine_text_under_n_chars=2000,
                    image_output_dir_path=output_path,
                    # strategy="fast"
)
                    
                    for element in raw_pdf_elements:
                        if 'CompositeElement' in str(type(element)):
                            # if element.metadata.page_number in l:
                            #     text = text + element.text
                            # else:
                            #     if text!=" \n ":
                            #         text_elements.append(Document(page_content=text,metadata={"source":pdf.name,"page_number":l[-1]}))
                            #     text=element.text
                            #     l.append(element.metadata.page_number)
                            text = text + element.text
                        elif 'Table' in str(type(element)):
                            summary = summarize_table(llm = ChatOpenAI(model="gpt-4-1106-preview",temperature=0),table_element=element.text)
                            text = text+summary
                    
                    text_elements.append(Document(page_content=text,metadata={"Document_name":pdf.name}))
                    
                
                    for image_file in os.listdir(output_path):
                        if image_file.endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(output_path, image_file)
                            encoded_image = encode_image(image_path)
                            image_elements.append(Document(page_content=encoded_image))
                    for i, ie in enumerate(image_elements):
                        summary = summarize_image(ie.page_content,llm=ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024,temperature=0)
)
                        text_elements.append(Document(page_content=summary,metadata={"Document_name":pdf.name,"image":"True"}))
                    
                    
                
                print(text_elements)
                tools = process_text(text_elements,image_elements)
                st.session_state.tools = tools
                chain = initialize_components(st.session_state.tools)
                st.session_state.chain = chain

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    display_conversation_history()

    if prompt := st.chat_input("Ask me anything about the uploaded documents."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        pdf_files = st.session_state.uploaded_files
        user_question = prompt.strip()

        if pdf_files and user_question:
            # st.session_state.messages = [message for message in st.session_state.messages if
            #                              message["role"] != "assistant"]
            handle_user_interaction(pdf_files, user_question)

            # for message in st.session_state.messages:
            #     if message["role"] == "assistant":
            #         with st.chat_message("assistant"):
            #             st.markdown(message["content"])

if __name__ == '__main__':
    main()