from crewai import Agent, Crew, Process, Task
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

import os
import agentops

load_dotenv()
agentops.init()

# --- Language Models ---
# [Azure OpenAI Language Model](https://github.com/crewAIInc/crewAI-examples/blob/main/azure_model/main.py)
# [Microsoft Docs](https://azure.microsoft.com/en-us/services/cognitive-services/openai/)

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

azure_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    model="gpt-4o",
    deployment_name="gpt-4o",
    temperature=0,
)

default_llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=f"{API_VERSION}",
    azure_endpoint=f"{RESOURCE_ENDPOINT}",
    deployment_name="gpt-4o",
    temperature=0,
)

# --- Tools ---
# PDF SOURCE: https://www.gpinspect.com/wp-content/uploads/2021/03/sample-home-report-inspection.pdf
pdf_search_tool = PDFSearchTool(
    pdf="./home_inspection_report.pdf",
    config=dict(
        # llm=dict(
        #     provider="azure_openai",  # or google, openai, anthropic, llama2, ...
        #     config=dict(
        #         model="gpt-4o",
        #         # temperature=0.5,
        #         # top_p=1,
        #         # stream=true,
        #     ),
        # ),
        # embedder=dict(
        #     provider="azure_openai",  # or openai, ollama, ...
        #     config=dict(
        #         model="text-embedding-ada-002",
        #         # deployment_name="gpt-4o",
        #         deployment_name="text-embedding-ada-002",
        #         # title="Embeddings",
        #     ),
        # ),
    ),
)


# --- Agents ---
research_agent = Agent(
    llm=default_llm,
    role="Research Agent",
    goal="Search through the PDF to find relevant answers",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The research agent is adept at searching and 
        extracting data from documents, ensuring accurate and prompt responses.
        """
    ),
    tools=[pdf_search_tool],
)


professional_writer_agent = Agent(
    llm=default_llm,
    role="Professional Writer",
    goal="Write professional emails based on the research agent's findings",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The professional writer agent has excellent writing skills and is able to craft 
        clear and concise emails based on the provided information.
        """
    ),
    tools=[],
)


# --- Tasks ---
answer_customer_question_task = Task(
    description=(
        """
        Answer the customer's questions based on the home inspection PDF.
        The research agent will search through the PDF to find the relevant answers.
        Your final answer MUST be clear and accurate, based on the content of the home
        inspection PDF.

        Here is the customer's question:
        {customer_question}
        """
    ),
    expected_output="""
        Provide clear and accurate answers to the customer's questions based on 
        the content of the home inspection PDF.
        """,
    tools=[pdf_search_tool],
    agent=research_agent,
)

write_email_task = Task(
    description=(
        """
        - Write a professional email to a contractor based 
            on the research agent's findings.
        - The email should clearly state the issues found in the specified section 
            of the report and request a quote or action plan for fixing these issues.
        - Ensure the email is signed with the following details:
        
            Best regards,

            Brandon Hancock,
            Hancock Realty
        """
    ),
    expected_output="""
        Write a clear and concise email that can be sent to a contractor to address the 
        issues found in the home inspection report.
        """,
    tools=[],
    agent=professional_writer_agent,
)

# --- Crew ---
crew = Crew(
    agents=[research_agent, professional_writer_agent],
    tasks=[answer_customer_question_task, write_email_task],
    process=Process.sequential,
    llm=azure_llm,
    # embedder={
    #     "provider": "azure_openai",
    #     "config": {
    #         # model": "gpt-4o",
    #         # "deployment_name": "text-embedding-3-large",
    #         "deployment_name": "gpt-4o",
    #     },
    # },
)

customer_question = input(
    "Which section of the report would you like to generate a work order for?\n"
)
result = crew.kickoff_for_each(inputs=[{"customer_question": "Roof"}])
print(result)
