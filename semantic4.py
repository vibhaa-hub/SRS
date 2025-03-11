# Has req desc stakeholders AC without FS only struct along with txt file gen and SL sidebar
import streamlit as st
from langchain.prompts import PromptTemplate
from docx import Document as dox
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import os
from datetime import datetime

# Initialize model and prompt template
model = OllamaLLM(model="llama3.1:8b")

prompt_template = '''
<s>[INST]
You are a researcher tasked with answering questions about a SRS. All output must be in valid JSON. Don't add explanation beyond the JSON.
Please ensure your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer, please don't share false information.

Output answer in valid JSON using the following format: 
{{"requirement": requirement, "description": description, "stakeholders":[stakeholders],
   "user_stories": ["role": role,"story":user_story,"acceptance_criteria":[acceptance_criteria]]
   }}

What are user stories and acceptance criteria for the requirement description extracted from the document focusing on identifying different stakeholder perspectives? 
 [/INST]

[
 {{
    "requirement": "<Type of Requirement: functional/performance/security/usability/etc.>",
    "description": "<Detailed explanation of the requirement from the document>",
    "stakeholders": [
        "<Stakeholder 1>",
        "<Stakeholder 2>",
        "<Stakeholder 3>"
    ],
    "user_stories": [
        {{
            "role": "<Stakeholder Role>",
            "story": "As a [role], I want [goal] so that [benefit/value]",
            "acceptance_criteria": [
                "<Specific, measurable condition 1>",
                "<Specific, measurable condition 2>",
                "<Specific, measurable condition 3>"
            ]
        }},
        {{
            "role": "<Another Stakeholder Role>",
            "story": "As a [role], I want [goal] so that [benefit/value]",
            "acceptance_criteria": [
                "<Specific, measurable condition 1>",
                "<Specific, measurable condition 2>",
                "<Specific, measurable condition 3>"
            ]
        }}
    ]
}}

</s>

[INST]

What are user stories and acceptance criteria for the requirement description extracted from the document focusing on identifying different stakeholder perspectives? 
Output answer in valid JSON using the following format:
{{"requirement": requirement, "description": description, "stakeholders":[stakeholders],
   "user_stories": ["role": role,"story":user_story,"acceptance_criteria":[acceptance_criteria]]
   }}
SRS: {requirement}

[/INST]
'''

prompt = PromptTemplate(input_variables=["requirement"], template=prompt_template)
chain = prompt | model


def extract_requirements_from_docx(docx_file_path):
    document = dox(docx_file_path)
    requirements = []
    embedding = OllamaEmbeddings(model="nomic-embed-text")

    text_splitter = SemanticChunker(
        embeddings=embedding,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90.0
    )

    full_text = "\n".join([para.text.strip() for para in document.paragraphs if para.text.strip()])
    split_requirements = text_splitter.create_documents([full_text])

    requirements = [doc.page_content for doc in split_requirements if len(doc.page_content.strip()) > 50]
    print(f"Total number of chunks: {len(split_requirements)}")

    return requirements


def create_faiss_index(requirements, embedding):
    docs = [Document(page_content=req) for req in requirements]
    vector_store = FAISS.from_documents(docs, embedding)
    return vector_store


def retrieve_requirement(query, vector_store, k=1):
    docs = vector_store.similarity_search(query, k=k)
    return docs[0].page_content if docs else None


def generate_user_story(requirement):
    return chain.invoke(requirement)


def process_srs_and_print_user_stories(docx_file_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_directory, f"user_stories_{timestamp}.txt")

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        requirements = extract_requirements_from_docx(docx_file_path)
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        faiss_index = create_faiss_index(requirements, embedding)

        for i, requirement in enumerate(requirements):
            relevant_requirement = retrieve_requirement(requirement, faiss_index)
            user_story = generate_user_story(requirement)

            output_file.write(f"Chunk {i}:\n")
            output_file.write(f"Generated User Story:\n{user_story}\n")
            output_file.write("-" * 50 + "\n")

    return output_file_path


# # Streamlit Interface
# st.title('SRS to User Stories Generator')

# uploaded_file = st.file_uploader("Upload your SRS document", type=["docx"])

# if uploaded_file:
#     st.write("File uploaded successfully!")

#     output_directory = "D:/Code/OUTPUT"
#     if st.button("Generate User Stories"):
#         with st.spinner('Processing...'):
#             output_file_path = process_srs_and_print_user_stories(uploaded_file, output_directory)
#             st.success(f"User stories generated successfully! You can download the result from the link below.")
#             st.download_button(
#                 label="Download User Stories",
#                 data=open(output_file_path, 'rb'),
#                 file_name=f"user_stories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
#                 mime="text/plain"
#             )

# Streamlit Interface with improved UI
st.set_page_config(page_title="SRS to User Stories Generator", page_icon="📄", layout="wide")

# Sidebar for navigation and file upload
with st.sidebar:
    st.header("SRS to User Stories Generator")
    st.subheader("Upload your SRS document (in .docx format)")

# Main title and description
st.title("SRS to User Stories Generator 📄")
st.markdown("""
This tool allows you to upload an SRS (Software Requirements Specification) document and generates **user stories** and **acceptance criteria**.
Simply upload the document, and click the button below to generate the user stories.
""")

# File uploader with better UI
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
            output_file_path = process_srs_and_print_user_stories(uploaded_file, output_directory)
            
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
