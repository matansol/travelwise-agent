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


def parse_user_input(user_input: str) -> AIMessage:
    """
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


def query_qdrant(collection_name: str, customer_profile: str) -> list[str]:
    """ Query the Qdrant database to find options in the specified collection matching the customer profile.
        :param collection_name: The name of the collection to query (e.g., "flights", "hotels", "activities").
        :param customer_profile: The customer profile JSON object containing the key features.
    """

    query = f"Find options in {collection_name} matching this profile: {customer_profile}"
    vec = embedding_model.encode(query)
    results = qdrant_client.search(collection_name=collection_name, query_vector=vec, limit=5)
    return [result.payload for result in results]


def get_trip_options(user_input: str) -> str:
    """
    Get trip options based on the user input by querying Qdrant collections for flights, hotels, and activities.
    :param user_input: The user input describing their travel preferences.
    """
    customer_profile = parse_user_input(user_input)

    # Query Qdrant collections for flights, hotels, and activities
    flights_options = query_qdrant("flights", customer_profile)
    hotels_options = query_qdrant("hotels", customer_profile)
    activities_options = query_qdrant("activities", customer_profile)

    # Combine all options into a single list
    all_options = {"flights": flights_options, "hotels": hotels_options, "activities": activities_options}

    # Use LLM to assemble full trip options
    system_prompt = SystemMessage(
        content="You are a professional travel agent. Based on the user's trip request and the available flights, hotels, and activities, build the client 3 options for a trip. Please include both the incoming and return flights (they can be from different airlines and locations), the hotels, and the activities. Make sure to include the total price for each option.",)
    formatted_prompt = prompt_template.format(
        input=f"customer_profile: {customer_profile}, options: {all_options}",)
    messages = [system_prompt, HumanMessage(content=formatted_prompt)]

    # Get the trip options from the LLM
    response = llm(messages=messages)
    write_token_usage_to_csv(response)
    return response.content


def run_examples():
    """ Run the pipeline for the example input files and save the trip options to the corresponding output files. """
    examples_folder = "examples"

    # Get all files that start with "input" and end with ".txt"
    input_files = sorted(f for f in os.listdir(examples_folder) if f.startswith("input") and f.endswith(".txt"))
    for input_file in input_files:
        file_path = os.path.join(examples_folder, input_file)
        with open(file_path, "r") as file:
            user_input = file.read().strip()
        print(f"User Input from {input_file}: {user_input}")
        # Process input and get trip options
        trip_options = get_trip_options(user_input)

        # Create the corresponding output file
        output_file = input_file.replace("input", "output")  # e.g., input1.txt → output1.txt
        output_path = os.path.join(examples_folder, output_file)
        with open(output_path, "w") as file:
            file.write(str(trip_options))
        print(f"Trip options saved to {output_file}")
        print("\n" + "=" * 50 + "\n")


def run_pipeline():
    """ Run the TravelWise Agent pipeline to interactively get trip options based on user input. """
    hello_message = """\
    Welcome to the TravelWise Agent!
    
    I am your AI-driven assistant, here to help you plan your trips efficiently 
    and effectively. I can assist you with creating customized travel plans 
    based on your preferences, budget, and schedule.
    
    To get started, please share your travel preferences, such as your desired 
    destination, travel dates, budget, number of travelers, and any specific 
    interests or requirements. Based on this information, I will provide you 
    with tailored travel options and recommendations.
    """

    print(hello_message)
    user_input = input("Please describe your trip preferences: ").strip()
    print("\nThank you! Processing your travel preferences...\n")
    # Process input and get trip options
    trip_options = get_trip_options(user_input)
    print(trip_options)


if __name__ == "__main__":
    # run_examples()
    run_pipeline()
