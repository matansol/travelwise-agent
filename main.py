import os
import csv
from dotenv import load_dotenv
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.tools.python.tool import PythonTool
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI
import tiktoken

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OpenAI_API_KEY")

# Initialize OpenAI model
llm = OpenAI(api_key=API_KEY, temperature=0.7)

# Initialize Qdrant vector database
VECTOR_DB_URL = "https://example-qdrant-url.com"
VECTOR_DB_NAME = "travelwise"
vectorstore = Qdrant(VECTOR_DB_URL, VECTOR_DB_NAME)

# Initialize token counters
input_token_counter = 0
output_token_counter = 0

def count_tokens(text, is_input=True):
    global input_token_counter, output_token_counter
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    token_count = len(tokens)
    
    if is_input:
        input_token_counter += token_count
    else:
        output_token_counter += token_count
    
    return token_count

# Micro Agent 1 - Find Destination
def find_destination(customer_input, vectorstore):
    query = f"Find suitable destinations for: {customer_input}"
    count_tokens(query)
    results = vectorstore.similarity_search(query, k=5)
    return [result['text'] for result in results]

find_destination_tool = PythonTool(
    name="FindDestination",
    func=find_destination,
    description="Finds suitable travel destinations based on customer preferences and trends."
)

# Micro Agent 2 - Trip Matching
def trip_matching(destination, customer_profile, vectorstore):
    query = f"Find travel packages for {destination} that match {customer_profile}"
    count_tokens(query)
    results = vectorstore.similarity_search(query, k=5)
    return [result['text'] for result in results]

trip_matching_tool = PythonTool(
    name="TripMatching",
    func=trip_matching,
    description="Matches customer preferences with available itineraries."
)

# Micro Agent 3 - Commission Optimization
def optimize_commission(itineraries):
    optimized = sorted(itineraries, key=lambda x: x.get("commission_rate", 0), reverse=True)
    return optimized[:3]

commission_optimization_tool = PythonTool(
    name="CommissionOptimization",
    func=optimize_commission,
    description="Optimizes travel packages to maximize agent commissions."
)

# Define the tools
tools = [
    Tool(
        name="Find Destination",
        func=find_destination_tool.func,
        description=find_destination_tool.description,
    ),
    Tool(
        name="Trip Matching",
        func=trip_matching_tool.func,
        description=trip_matching_tool.description,
    ),
    Tool(
        name="Commission Optimization",
        func=commission_optimization_tool.func,
        description=commission_optimization_tool.description,
    ),
]

# Initialize the agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Interaction logic
def travelwise_agent():
    print("Welcome to TravelWise! I'm here to help you plan and book your trips.")
    while True:
        user_input = input("How can I assist you today? (type 'exit' to quit)\n")
        if user_input.lower() == "exit":
            print("Thank you for using TravelWise. Have a great day!")
            break
        count_tokens(user_input)
        response = agent.run(user_input)
        count_tokens(response, is_input=False)
        print(f"\n{response}\n")

# Save token usage to CSV
def save_token_usage():
    with open('token_count.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Total Input Tokens Used", "Total Output Tokens Used"])
        writer.writerow([input_token_counter, output_token_counter])

# Main execution
if __name__ == "__main__":
    travelwise_agent()
    save_token_usage()