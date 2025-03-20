import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


gherkin_prompt_template = """
Given the following software requirements specification (SRS) in natural language, 
generate Gherkin-based user stories with 'Given', 'When', 'Then' steps for each SRS requirement. Format each user story in a separate block.

SRS Requirement:
{requirement}
"""

step_definitions_prompt_template = """
Given the following Gherkin user story, 
generate Python step definitions for a BDD framework like Behave, with `@given`, `@when`, and `@then` decorators. Format each step definition in a separate block.

Gherkin User Story:
{gherkin_user_story}
"""

gherkin_prompt = PromptTemplate(input_variables=["requirement"], template=gherkin_prompt_template)
step_definitions_prompt = PromptTemplate(input_variables=["gherkin_user_story"], template=step_definitions_prompt_template)

os.environ["GOOGLE_API_KEY"] = "IzaSyBC_TAoSPKWHDsqKDcWR8S2ai9tITCyHH"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

gherkin_chain = gherkin_prompt | llm
step_definitions_chain = step_definitions_prompt | llm

def generate_gherkin_and_steps_for_multiple_scenarios(srs_input):
    requirements = srs_input.split("\n")
    
    for requirement in requirements:
        if requirement.strip():

            gherkin_result = gherkin_chain.invoke(requirement.strip())
            
            st.subheader(f"Generated Gherkin User Story for: {requirement.strip()}")
            st.code(gherkin_result.content, language="gherkin")
            
            step_definitions_result = step_definitions_chain.invoke(gherkin_result.content)

            st.subheader(f"Generated Python Step Definitions for: {requirement.strip()}")
            st.code(step_definitions_result.content, language="python")


st.title("Gherkin User Stories and Step Definitions Generator")

srs_input = """
The system shall allow user to confirm the purchase.

The system shall enable user to enter the payment information.

The system shall provide a uniform look and feel between all the web pages.

The system shall provide a digital image for each product in the product catalog.

The system shall provide use of icons and toolbars.

"""

if st.button("Generate Gherkin User Stories and Step Definitions"):
    generate_gherkin_and_steps_for_multiple_scenarios(srs_input)
