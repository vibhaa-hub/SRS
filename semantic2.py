# Has req desc stakeholders AC with FS along with txt file gen
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


# model = CTransformers(model=model_path, temperature=0.0, config={'max_new_tokens': 9000, 'context _length': 9000})
model=OllamaLLM(model="llama3.1:8b")

prompt_template='''
<s>[INST]
You are a researcher tasked with answering questions about a SRS. All output must be in valid JSON. Don't add explanation beyond the JSON.
Please ensure your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer, please don't share false information.

Output answer in valid JSON using the following format: 
{{"requirement: requirement, "description": description, "stakeholders":[stakeholders],
   "user_stories": ["role": role,"story":user_story,"acceptance_criteria":[acceptance_criteria]]
   }}

What are user stories and acceptance criteria for the requirement description extracted from the document focusing on identifying different stakeholder perspectives? 
 [/INST]
{{
    [
        {{
            "requirement": "Functional",
            "description": "The system shall display all the available components of the product to configure.",
            "stakeholders": [
                "Product Users",
                "System Administrators", 
                "Configuration Managers"
            ],
            "user_stories": [
                {{
                    "role": "Product User",
                    "story": "As a product user, I want to see all available components so that I can configure the product according to my needs",
                    "acceptance_criteria": [
                        "The system displays a list of all available components",
                        "Each component has a clear description and image",
                        "Users can filter components by category"
                    ]
                }},
                {{
                    "role": "System Administrator",
                    "story": "As a system administrator, I want a comprehensive view of all configurable components so that I can manage and oversee system configuration options",
                    "acceptance_criteria": [
                        "Administrators can view all system components",
                        "Detailed technical specifications are available for each component",
                        "Administrators can set default configuration parameters"
                    ]
                }},
                {{
                    "role": "Configuration Manager",
                    "story": "As a configuration manager, I want a centralized view of all product components so that I can ensure consistent and standardized configurations",
                    "acceptance_criteria": [
                        "Configuration managers can see all component variations",
                        "Ability to create and manage configuration templates",
                        "Tracking of configuration changes and version history"
                    ]
                }}
            ]
        }},
        {{
            "requirement": "Functional",
            "description": "The system shall notify the user about any conflict in the current configuration.",
            "stakeholders": [
                "Product Users",
                "System Administrators",
                "Configuration Managers"
            ],
            "user_stories": [
                {{
                    "role": "Product User",
                    "story": "As a product user, I want to be notified of any conflicts in my configuration so that I can resolve them promptly",
                    "acceptance_criteria": [
                        "The system provides real-time notifications for configuration conflicts",
                        "Notifications include details about the conflict and suggested resolutions",
                        "Users can acknowledge and dismiss notifications"
                    ]
                }},
                {{
                    "role": "System Administrator",
                    "story": "As a system administrator, I want detailed conflict notifications so that I can quickly identify and mitigate configuration issues",
                    "acceptance_criteria": [
                        "Comprehensive conflict reports are generated",
                        "Conflict severity levels are clearly indicated",
                        "Automated suggestions for conflict resolution are provided"
                    ]
                }},
                {{
                    "role": "Configuration Manager",
                    "story": "As a configuration manager, I want a log of all configuration conflicts so that I can analyze patterns and improve configuration guidelines",
                    "acceptance_criteria": [
                        "Detailed conflict logs are maintained",
                        "Historical conflict data can be exported and analyzed",
                        "Ability to create configuration rules to prevent recurring conflicts"
                    ]
                }}
            ]
        }},
        {{
            "requirement": "Performance",
            "description": "The system shall provide a contractual agreement with an internet service provider for T3 access with 99.9999% availability.",
            "stakeholders": [
                "System Administrators",
                "IT Infrastructure Team",
                "Executive Management"
            ],
            "user_stories": [
                {{
                    "role": "System Administrator",
                    "story": "As a system administrator, I want to ensure that our internet service provider guarantees high availability so that our services remain uninterrupted",
                    "acceptance_criteria": [
                        "A formal contract is in place with the internet service provider",
                        "The contract specifies 99.9999% availability",
                        "Performance metrics are monitored and reported monthly"
                    ]
                }},
                {{
                    "role": "IT Infrastructure Team",
                    "story": "As an IT infrastructure team member, I want detailed performance reports so that I can verify and maintain our high availability",
                                        "acceptance_criteria": [
                        "Detailed network performance dashboards are available",
                        "Real-time monitoring of network uptime and latency",
                        "Automated alerts for any performance degradation"
                    ]
                }},
                {{
                    "role": "Executive Management",
                    "story": "As an executive, I want comprehensive visibility into our system's availability and performance so that I can make informed strategic decisions",
                    "acceptance_criteria": [
                        "Monthly executive reports on system availability",
                        "Comparative analysis of performance against contractual SLAs",
                        "Financial impact assessments of any downtime"
                    ]
                }}
            ]
        }},
        {{
            "requirement": "Security",
            "description": "The system shall use secure sockets in all transactions that include any confidential customer information.",
            "stakeholders": [
                "Customers",
                "Security Team",
                "Compliance Officers"
            ],
            "user_stories": [
                {{
                    "role": "Customer",
                    "story": "As a customer, I want my confidential information to be transmitted securely so that I can trust the system with my data",
                    "acceptance_criteria": [
                        "All transactions involving confidential information use secure sockets (SSL/TLS)",
                        "Users are informed when secure connections are established",
                        "Security certificates are valid and up to date"
                    ]
                }},
                {{
                    "role": "Security Team",
                    "story": "As a security team member, I want to implement and monitor robust encryption protocols so that we can protect customer data from potential breaches",
                    "acceptance_criteria": [
                        "Implementation of end-to-end encryption",
                        "Regular security audits and penetration testing",
                        "Detailed logging of all security-related events"
                    ]
                }},
                {{
                    "role": "Compliance Officer",
                    "story": "As a compliance officer, I want to ensure that our data transmission meets all regulatory requirements so that we maintain legal and industry standards",
                    "acceptance_criteria": [
                        "Compliance with GDPR, HIPAA, and other relevant data protection regulations",
                        "Comprehensive documentation of security measures",
                        "Annual compliance certification and reporting"
                    ]
                }}
            ]
        }}
    ]
}}

</s>

[INST]

What are user stories and acceptance criteria for the requirement description extracted from the document focusing on identifying different stakeholder perspectives? 
Output answer in valid JSON using the following format:
{{"requirement: requirement, "description": description, "stakeholders":[stakeholders],
   "user_stories": ["role": role,"story":user_story,"acceptance_criteria":[acceptance_criteria]]
   }}
SRS: {requirement}

[/INST]
'''
prompt =PromptTemplate(input_variables=["requirement"],template =prompt_template)

chain=prompt|model

def extract_requirements_from_docx(docx_file_path):

 document =dox(docx_file_path)
 requirements =[]
 embedding = OllamaEmbeddings(model="nomic-embed-text")

#text_splitter = RecursiveCharacterTextSplitter(chunk size=1000.chunk_overlap 100.separators= ["\n\n\n","\n\n"."\n"""" "length_function=len)
#text_splitter = SemanticChunker(OllamaEmbeddings(model_name=embedding,model_kwargs={'device':'cpu'},encoding_kwargs={'normalize_embeddings': True},
 text_splitter = SemanticChunker(embeddings=embedding,   
                                breakpoint_threshold_type="percentile",
                                breakpoint_threshold_amount=90.0)

 full_text="\n".join([para.text.strip() for para in document.paragraphs if para.text.strip()])

 split_requirements = text_splitter.create_documents([full_text])

 requirements = [doc.page_content for doc in split_requirements if len(doc.page_content.strip()) > 50]
 print(f"Total number of chunks: {len(split_requirements)}")

#  for i, chunk in enumerate(requirements[:5], 1):
#   print(f"Chunk {i}: {chunk[:500]}...")

 return requirements

def create_faiss_index(requirements, embedding):
 docs=[Document(page_content=req) for req in requirements]
 vector_store=FAISS.from_documents(docs, embedding)
 return vector_store

def retrieve_requirement(query,vector_store, k=1):
 docs=vector_store.similarity_search(query,k=k)
 return docs[0].page_content if docs else None

def generate_user_story(requirement):
 return chain.invoke(requirement)

def process_srs_and_print_user_stories(docx_file_path,output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_directory, f"user_stories_{timestamp}.txt")
    
    # Open the output file in write mode
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
     requirements = extract_requirements_from_docx(docx_file_path)
     embedding = OllamaEmbeddings(model="nomic-embed-text")
     faiss_index=create_faiss_index(requirements, embedding)
     for i, requirement in enumerate(requirements):
      relevant_requirement =retrieve_requirement(requirement,faiss_index)
      user_story = generate_user_story(requirement)
      # Write to file
      # output_file.write(f"Requirement {i}: {requirement}\n")
      output_file.write(f"Chunk {i}:\n")
      output_file.write(f"Generated User Story:\n{user_story}\n")
      output_file.write("-"*50 + "\n")
    
    print(f"User stories have been written to {output_file_path}")


SRS_DOC_PATH ="C:/Users/vibha/Downloads/SRS4.0.docx"
OUTPUT_DIRECTORY = "D:/Code/OUTPUT"
process_srs_and_print_user_stories(SRS_DOC_PATH, OUTPUT_DIRECTORY)