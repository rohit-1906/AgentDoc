import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.vectorstores import FAISS,Chroma
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer, util
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.chains import LLMChain
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent,ZeroShotAgent
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
# Set up LLM caching using an in-memory cache
set_llm_cache(InMemoryCache())

# Title for the Streamlit app
st.title("PDF Summarizer, QA & Chat")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

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
    def initialize_components(tools):

        OpenAIModel = "gpt-4"
        #memory = ConversationBufferMemory(memory_key="conversation_history")
        
        llm = ChatOpenAI(model=OpenAIModel, temperature=0, openai_api_key=load_openai_api_key())
               
        prefix = """Have a conversation with a human, answering the following questions as best you can. You should only use the retrieval tool and not any other: 
                    Reminder : You have the ability to answer questions related to the document If a question is not related to the provided document or falls outside general topics. Feel free to start with greetings or general queries about the document. """
        suffix = """Begin!" 
        
        {chat_history}

        Question: {input}

        {agent_scratchpad}"""


        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        print(prompt)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        memory = ConversationBufferMemory(memory_key="chat_history")
        agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory,handle_parsing_errors=True,max_iterations=10)


        return agent_chain

    # Perform initialization and store the chain in session state
    
    # print("Initialization complete.")

# process text extracted from PDFs and create a knowledge base
    



def process_text(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=200
    )
    # chunks = text_splitter.split_documents(doc)
    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # knowledgeBase = FAISS.from_documents(chunks, embeddings)
    
    # retriever = knowledgeBase.as_retriever(search_type="mmr",
    #             search_kwargs={'k': 7, 'fetch_k': 50})
    vectorstore = Chroma(embedding_function = embeddings)
    retriever = ParentDocumentRetriever(
        docstore=InMemoryStore(),
        vectorstore=vectorstore,
        parent_splitter=text_splitter,
        child_splitter=child_text_splitter
    )
    retriever.add_documents(doc)
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
    tool = create_retriever_tool(
    compression_retriever,
    "document_retrieval_tool",
    "Retrieves relevant chunks from the document based on user question .",
)
    tools = [tool]
    return tools

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
    agent_executor = st.session_state.chain

    response = agent_executor.run(input=user_question)
    assistant_response = response
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
def main():
    with st.sidebar.expander("Upload your PDF Documents"):
        pdf_files = st.sidebar.file_uploader(' ', type='pdf', accept_multiple_files=True)
        if pdf_files:
            st.session_state.uploaded_files = pdf_files
            if "tools" not in st.session_state:
                doc = []
                for pdf in pdf_files:
                    pdf_reader = PdfReader(pdf)
                    i=0
                    for page in pdf_reader.pages:
                        i=i+1
                        text = page.extract_text()
                        doc.append(Document(page_content=text,metadata={"source":pdf.name,"page":i}))
                    

                tools = process_text(doc)
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