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
from langchain_openai import OpenAIEmbeddings
from langchain.tools import BaseTool
from functools import partial
from typing import Optional
from langchain.tools import Tool
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, format_document
from langchain_core.pydantic_v1 import BaseModel, Field





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
    def initialize_components(tools):

        OpenAIModel = "gpt-4-1106-preview"
        #memory = ConversationBufferMemory(memory_key="conversation_history")
        
        llm = ChatOpenAI(model=OpenAIModel, temperature=0, openai_api_key=load_openai_api_key())
               
        prefix = """
        Your role is to assist in extracting information from documents to answer questions accurately. You have two powerful tools at your disposal:

        1. Document Retriever: This tool excels at comprehending and extracting information from the text content of various documents. Employ it whenever the question pertains to the document's context, such as:

        Key findings or themes
        Specific details or facts
        Explanations of concepts or events
        Relationships between different parts of the document
        
        2. Metadata Retriever: This tool specializes in retrieving information about the document itself, such as its Document name , Page. Utilize it when the question focuses on these details, for example:

        Finding a document by its name or author
        Determining the date a document was created
        Locating specific information within a document based on page numbers or sections
        Your task is to carefully analyze each question and determine the most appropriate tool to employ for retrieving the most relevant and accurate answer.

        Additionally, a summary logic is available to condense and present key insights derived from the documents. If the question involves obtaining an overview or summary of the document content, leverage this logic for a concise response."""

        suffix = """
        If the user queries you to do a task beyond your objectives, you must politely decline and proceed to `Final Answer` where you suggest to ask questions related to context of documents uploaded.
        Remember:

        Craft your queries in a way that maximizes the effectiveness of the chosen tool.
        Strive for clarity and precision in your responses.
        Aim to provide the most informative and helpful answers possible, drawing upon the strengths of the available tools and logic.
        Begin!

        {chat_history}

        Question: {input}

        {agent_scratchpad}
        
        Note: You should only use the tools . You should not answer the Question which is not in the Document"""

        
        # print(prompt)
        # llm_chain = LLMChain(llm=llm, prompt=prompt)

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        # print(prompt)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        memory = ConversationBufferMemory(memory_key="chat_history")
        agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory,handle_parsing_errors=True,max_iterations=10,early_stopping_method='generate',max_execution_time=5)


        return agent_chain

    # Perform initialization and store the chain in session state
    
    # print("Initialization complete.")

# process text extracted from PDFs and create a knowledge base



def process_text(doc,image_elements):

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
    # print(doc)
    retriever.add_documents(doc)
    # compressor = CohereRerank()
    # compression_retriever = ContextualCompressionRetriever(
    # base_compressor=compressor, base_retriever=retriever
# )

    class RetrieverInput(BaseModel):
        """Input to the retriever."""

        query: str = Field(description="query to look up in retriever")
    def _get_relevant_documents(
        query: str,
        retriever: retriever,
        document_prompt: BasePromptTemplate,
        document_separator: str,
    ) -> str:
        docs = retriever.get_relevant_documents(query)
        return document_separator.join(
            format_document(doc, document_prompt) for doc in docs
        )


    async def _aget_relevant_documents(
        query: str,
        retriever: retriever,
        document_prompt: BasePromptTemplate,
        document_separator: str,
    ) -> str:
        docs = await retriever.aget_relevant_documents(query)
        return document_separator.join(
            format_document(doc, document_prompt) for doc in docs
        )


    def create_retriever_tool(
        retriever: retriever,
        name: str,
        description: str,
        *,
        document_prompt: Optional[BasePromptTemplate] = None,
        document_separator: str = "\n\n",
    ) -> Tool:
        """Create a tool to do retrieval of documents.

        Args:
            retriever: The retriever to use for the retrieval
            name: The name for the tool. This will be passed to the language model,
                so should be unique and somewhat descriptive.
            description: The description for the tool. This will be passed to the language
                model, so should be descriptive.

        Returns:
            Tool class to pass to an agent
        """
        document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")
        # print(document_prompt)
        func = partial(
            _get_relevant_documents,
            retriever=retriever,
            document_prompt=document_prompt,
            document_separator=document_separator,
        )
        afunc = partial(
            _aget_relevant_documents,
            retriever=retriever,
            document_prompt=document_prompt,
            document_separator=document_separator,
        )
        return Tool(
            name=name,
            description=description,
            func=func,
            coroutine=afunc,
            args_schema=RetrieverInput,
        )

        
    metadata_field_info = [
    AttributeInfo(
        name="source",
        description="its the name of the parent document from where the content is generated",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page number in which the content is present.",
        type="integer",
    ),]
    self_retriever = SelfQueryRetriever.from_llm(
        vectorstore=vectorstore,metadata_field_info=metadata_field_info,llm=ChatOpenAI(temperature=0),document_contents="Contains any type of data"
    )

#     tool1 = create_retriever_tool(
#     retriever,
#     "document_retrieval_tool",
#     "Retrieves relevant chunks from the document based on user question .",
# )
    # temp=[Document(page_content='self_retrieve(query) - use this tool when you need to answer the question about the document which is present in the metadata of chunks based on user question', metadata={'index': 0}), Document(page_content='document_retrieval_tool(query, retriever_tools=[]) - use this tool when you need to answer the question from the document based on user question', metadata={'index': 1})]

    # vectorstore_tools = FAISS.from_documents(temp, OpenAIEmbeddings())

    # retriever_tools = vectorstore_tools.as_retriever()

    tool1 = create_retriever_tool(
                retriever,
                "document_retrieval_tool",
                "Retrieves relevant chunks from the document based on user question ."
    )
    tool2 = create_retriever_tool(
                self_retriever,
                "Metadata_Retriever",
                "Retrieves detailed metadata information about documents."
    )
    class self_retrieve(BaseTool):
        name = "self_retrieval_tool"
        description = "use this tool when you need to answer the question about the document which is present in the metadata of chunks based on user question"

        def _run(self, query: str):
            retrieved_docs=self_retriever.get_relevant_documents(query)
            return retrieved_docs
        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")

    tools = [tool1,self_retrieve()]

    # tools = [tool1,tool2]
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
    rag_chain = st.session_state.chain
    response = rag_chain.invoke({"input": user_question, "chat_history": st.session_state.chat_history})
    # print(response)

    st.session_state.chat_history.extend([HumanMessage(content=user_question), response])
    assistant_response = response['output']
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
        output_path = os.path.join(os.getcwd(), "figures")
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
                    
                    # text_elements.append(Document(page_content=text,metadata={"Document_name":pdf.name}))
                        text_elements.append(Document(page_content=text,metadata={"Document_name":pdf.name,"page_no": element.metadata.page_number}))
                
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