import os
import sys
import warnings


import openai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.clf_travelwise import write_token_usage_to_csv
from src.env_variables import get_qdrant_credentials

warnings.simplefilter("ignore")

# Config
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # API key
DEPLOYMENT_NAME = "team4-gpt4o"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

# Set the OpenAI API key and endpoint
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = "azure"
openai.api_version = "2023-05-15"

# Initialize the Azure OpenAI Chat Model
llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=API_VERSION,
    openai_api_type="azure",
    temperature=0.7,
)

# Initialize the Qdrant database client
qdrant_url, qdrant_api_key = get_qdrant_credentials()
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Intialize embeddings model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # ('multi-qa-mpnet-base-dot-v1') - stronget model
VECTOR_DIM = embedding_model.get_sentence_embedding_dimension()


# Define a chat prompt template
prompt_template = ChatPromptTemplate.from_template("{input}")

def query_qdrant(collection_name: str, customer_profile: str) -> list[str]:
    """Query the Qdrant database to find options in the specified collection matching the customer profile.
    :param collection_name: The name of the collection to query (e.g., "flights", "hotels", "activities").
    :param customer_profile: The customer profile JSON object containing the key features.
    """

    query = f"The profile: {customer_profile}"
    vec = embedding_model.encode(query)
    results = qdrant_client.search(collection_name=collection_name, query_vector=vec, limit=5)
    return [result.payload for result in results]


def parse_user_input(user_input: str) -> AIMessage:
    """
    Micro Agent 1 - Parse User Input
    Parse user input to extract key features like budget, number of people,
    wanted time, activities, and former trips.
    :param user_input: The user input describing their travel preferences.
    """
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
        Extract the following information from the user input and return it as a JSON object, if the informtion is not present write "not specified":
        - Budget
        - Number of people
        - Desired duration
        - Preferred activities
        - Previous trips and their feedback

        User input: {user_input}
        """,
    )
    formatted_prompt = prompt.format(user_input=user_input)
    messages = [HumanMessage(content=formatted_prompt)]
    response = llm(messages=messages)
    return response


def output_parser(user_profile, rag_parser: str) -> AIMessage:
    """
    Micro Agent 3 - Output Parser
    Make sure that the output is valid and aligned with the customer profile.
    """
    output_conditions = "Is the output aligned with the customer profile? return 0 if not, otherwise return 1."
    response = use_preformance_monitor(user_profile, rag_parser, output_conditions)
    if int(response.content[0]) == 0:
        return rag_parser, 0
    else:
        return rag_parser, 1
    


def use_preformance_monitor(input, output, output_conditions):
    """
    Micro Agent 4 - Performance Monitor
    An agent that monitors the performance of the other agents by looking at the input and output and the conditions of the output.
    The agent make sure that the output is valid and that the conditions are met."""

    system_prompt = SystemMessage(content="You are a preformance monitor agent, based on the input and the output conditions, evaluate if the output is valid."
    "for example if the output is valid then return only the number 1 otherwise return only the numver 0.")
    formatted_prompt = prompt_template.format(
        input=f"input: {input}, output_conditions: {output_conditions}, output: {output}",
    )
    messages = [system_prompt, HumanMessage(content=formatted_prompt)]
    response = llm(messages=messages)
    return response