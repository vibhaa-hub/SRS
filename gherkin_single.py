import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st  

# Define the prompt template for Gherkin User Stories
gherkin_prompt_template = """
Given the following software requirements specification (SRS) in natural language, 
generate Gherkin-based user stories with 'Given', 'When', 'Then' steps for each SRS requirement. Format each user story in a separate block.

SRS Requirements:
{requirement}
"""

# Define the prompt template for Python Step Definitions
step_definitions_prompt_template = """
Given the following Gherkin user stories, 
generate Python step definitions for a BDD framework like Behave, with `@given`, `@when`, and `@then` decorators. Format each step definition in a separate block.

Gherkin User Stories:
{gherkin_user_stories}
"""

# Initialize the prompt templates with the required variable
gherkin_prompt = PromptTemplate(input_variables=["requirement"], template=gherkin_prompt_template)
step_definitions_prompt = PromptTemplate(input_variables=["gherkin_user_stories"], template=step_definitions_prompt_template)

# Set up Google API Key for ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = "AIzaSyBC_TAoSPKWHDsqKDcWR8S2ai9tITCyHHA"  # Replace with your Google API key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the chain for both prompts
gherkin_chain = gherkin_prompt | llm
step_definitions_chain = step_definitions_prompt | llm

# Function to generate Gherkin and Step Definitions
def generate_gherkin_and_steps(srs_input):
    # Generate Gherkin User Stories from the SRS input
    gherkin_result = gherkin_chain.invoke(srs_input)
    
    # Display the raw Gherkin User Stories
    print("Raw Output from Model (Gherkin User Stories):")
    print(gherkin_result.content)  # Print Gherkin User Stories
    st.subheader("Raw Output from Model (Gherkin User Stories):")
    st.code(gherkin_result.content, language='gherkin') 
    
    # Pass the Gherkin User Stories as input to the second prompt to generate Step Definitions
    step_definitions_result = step_definitions_chain.invoke(gherkin_result.content)
    
    # Display the raw Python Step Definitions
    print("\nRaw Output from Model (Python Step Definitions):")
    print(step_definitions_result.content)  # Print Step Definitions
    st.subheader("Raw Output from Model (Python Step Definitions):")
    st.code(step_definitions_result.content, language='python')

st.title("Gherkin User Stories and Step Definitions Generator")

# Hardcoded SRS input
srs_input = """
The system shall allow user to confirm the purchase.

The system shall enable user to enter the payment information.

The system shall provide a uniform look and feel between all the web pages.

The system shall provide a digital image for each product in the product catalog.

The system shall provide use of icons and toolbars.

"""

if st.button("Generate Gherkin User Stories and Step Definitions"):

# Generate Gherkin User Stories and Step Definitions
  generate_gherkin_and_steps(srs_input)
