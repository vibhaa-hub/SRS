# Using chaining
import os
from datetime import datetime
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup the model for both prompts

# model = OllamaLLM(model="llama3.1:8b")

os.environ["GOOGLE_API_KEY"] = "AIzaSyBC_TAoSPKWHDsqKDcWR8S2ai9tITCyHHA"
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# First prompt: Raw Text Output (Now includes requirement description)
prompt_template_first = '''
<s>[INST]
You are a researcher tasked with answering questions about an SRS document. All output must be in plain text, without any JSON formatting. Don't add explanations beyond the raw output.
Please ensure your responses are socially unbiased and positive in nature.
If you don't know the answer, please don't share false information.

Output only raw text with the following details: 
- Requirement Type: <Type of Requirement: functional/performance/security/usability/etc.>
- Requirement Description: <Provide a detailed explanation of the requirement from the document>
- Stakeholders: <List of stakeholders>
Example Output:
"Requirement Type: functional
Requirement Description: The system should allow users to log in using their email and password.
Stakeholders: user, developer, QA"

Extract the following from the document and give the output in raw text format.
Ensure that you do not truncate the response, and provide all the requirements
</INST>

SRS: {requirement}
'''

# Second prompt: User Stories and Acceptance Criteria (Takes raw text as input)
prompt_template_second = '''
<s>[INST]
You are a researcher tasked with generating user stories and acceptance criteria from the raw text that describes a requirement and its stakeholders. 
Please use the provided raw text and extract the necessary information to generate user stories and acceptance criteria for each stakeholder.
Format your response in raw text, and ensure that each stakeholder has corresponding user stories and acceptance criteria.

Input: {raw_text}

Output Example:
"As a user, I want to be able to log in so that I can access my account."
Acceptance Criteria:
1. User can log in with valid credentials.
2. Login screen is displayed correctly on all devices.
3. User is redirected to the homepage upon successful login."

Ensure that you do not truncate the response, and provide the full user stories and acceptance criteria for all stakeholders.
</INST>
'''

# Function to generate raw text output for requirements, type, and stakeholders
def generate_raw_text(requirement):
    prompt = PromptTemplate(input_variables=["requirement"], template=prompt_template_first)
    first_chain = prompt | model  # Use the LLM chain to generate raw text
    raw_text = first_chain.invoke(requirement).content # Get raw text from the first prompt
    return raw_text

# Function to generate user stories from raw text output
def generate_user_stories_from_raw_text(raw_text):
    prompt = PromptTemplate(input_variables=["raw_text"], template=prompt_template_second)
    second_chain = prompt | model  # Use the LLM chain to process raw text and generate user stories
    user_stories = second_chain.invoke(raw_text).content # Generate user stories based on raw text
    return user_stories

# Function to extract requirements from the document
def extract_requirements_from_docx(docx_file_path):
    from docx import Document as dox
    document = dox(docx_file_path)
    requirements = []

    full_text = "\n".join([para.text.strip() for para in document.paragraphs if para.text.strip()])
    requirements.append(full_text)

    return requirements

# Function to create FAISS index for requirements
def create_faiss_index(requirements, embedding):
    docs = [Document(page_content=req) for req in requirements]
    vector_store = FAISS.from_documents(docs, embedding)
    return vector_store

# Function to retrieve a relevant requirement from FAISS index
def retrieve_requirement(query, vector_store, k=1):
    docs = vector_store.similarity_search(query, k=k)
    return docs[0].page_content if docs else None

# Main function to process the SRS and generate user stories
def process_srs_and_generate_stories(docx_file_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_directory, f"user_stories_{timestamp}.txt")

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        requirements = extract_requirements_from_docx(docx_file_path)
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        faiss_index = create_faiss_index(requirements, embedding)

        for i, requirement in enumerate(requirements):
            relevant_requirement = retrieve_requirement(requirement, faiss_index)
            
            # Step 1: Generate raw text for requirement, type, stakeholders
            raw_text = generate_raw_text(relevant_requirement)
            
            # Step 2: Generate user stories from raw text output
            user_stories = generate_user_stories_from_raw_text(raw_text)

            output_file.write(f"Chunk {i}:\n")
            output_file.write(f"Raw Text: {raw_text}\n")
            output_file.write(f"Generated User Stories:\n{user_stories}\n")
            output_file.write("-" * 50 + "\n")

    return output_file_path


st.set_page_config(page_title="SRS to User Stories Generator", page_icon="📄", layout="wide")


with st.sidebar:
    st.header("SRS to User Stories Generator")
    st.subheader("Upload your SRS document (in .docx format)")


st.title("SRS to User Stories Generator 📄")
st.markdown("""
This tool allows you to upload an SRS (Software Requirements Specification) document and generates **user stories** and **acceptance criteria**.
Simply upload the document, and click the button below to generate the user stories.
""")

uploaded_file = st.file_uploader("Choose a `.docx` file", type=["docx"], label_visibility="collapsed")

if uploaded_file:
    st.success("File uploaded successfully! 🎉")

    # Option to choose the output directory
    output_directory = st.text_input("Output Directory", "D:/Code/OUTPUT", help="Specify the directory to save the generated file.")

    # Button to generate user stories
    if st.button("Generate User Stories 🔄"):
        with st.spinner('Processing... Please wait.'):

            # Show a progress bar
            progress_bar = st.progress(0)
            output_file_path = process_srs_and_generate_stories(uploaded_file, output_directory)
            
            for i in range(100):
                progress_bar.progress(i + 1)  # Simulating progress during processing
            
            st.success("User stories generated successfully! 🎉")

            # Provide download link for the output file
            st.markdown("#### Download the user stories below:")
            st.download_button(
                label="Download User Stories 📥",
                data=open(output_file_path, 'rb'),
                file_name=f"user_stories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
else:
    st.warning("Please upload an SRS document to get started.")
