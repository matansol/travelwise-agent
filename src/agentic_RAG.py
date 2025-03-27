from src.micro_agents import *
import ast


def choose_tools(customer_profile: str) -> SystemMessage:
    """
    Based on the customer profile, choose the tools that the RAG should use to build the trip.
    """
    # Use LLM to choose the best tools for the trip - what kind of data we need to pull from the database
    system_prompt = SystemMessage(
        content="You are a micro-agent within a workflow designed to generate a trip plan based on user preferences. "
        "Your role is to select the appropriate database sources for the RAG system. There are three database types: flights, activities, and hotels. "
        "Based on the user's prompt, determine the relevant collections (any combination of these options) and return only the list of selected collection names."
        "Output example: ['flights', 'hotels']")
    formatted_prompt = prompt_template.format(
        input=f"customer_profile: {customer_profile}",
    )
    messages = [system_prompt, HumanMessage(content=formatted_prompt)]
    response = llm(messages=messages)
    write_token_usage_to_csv(response)
    # Convert the string representation of a list to an actual list
    response_list = ast.literal_eval(response.content)
    return response_list

def get_database_data(customer_profile: str) -> str:
    # pull the rellevant data from the database to help the agent build the trip
    collection_names = choose_tools(customer_profile)
    all_options = {}
    # Find the best destinations based on the customer profile
    flight_system_prompt = SystemMessage(
        content="You are a professional travel agent. Based on the customer profile choose 2 options for destinations and a proper dates for the trip. \n"
        "give your answer in key words to be used to look for the best options in the database."
        "list of destinations: New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia.\n"
    )
    flight_formatted_prompt = prompt_template.format(
        input=f"customer_profile: {customer_profile}",
    )
    messages = [flight_system_prompt, HumanMessage(content=flight_formatted_prompt)]
    qdrant_query_destination = llm(messages=messages)
    write_token_usage_to_csv(qdrant_query_destination)

    if "flights" in collection_names:    
        flights_options = query_qdrant("flights", qdrant_query_destination.content)
        all_options["flights"] = flights_options

    if "hotels" in collection_names:
        hotels_system_prompt = SystemMessage(
            content="Based on the customer profile describe the type of hotel room that would best suit for his trip. \n"
            "list of options: single room, double room, suite, penthouse.\n"
        )
        hotels_formatted_prompt = prompt_template.format(
            input=f"customer_profile: {customer_profile}, destination options: {qdrant_query_destination.content}",
        )
        messages = [hotels_system_prompt, HumanMessage(content=hotels_formatted_prompt)]
        qdrant_query_hotels = llm(messages=messages)
        write_token_usage_to_csv(qdrant_query_hotels)
        hotels_options = query_qdrant("hotels", qdrant_query_hotels.content)
        all_options["hotels"] = hotels_options
    
    if "activities" in collection_names:
        activity_types = [
            "Museum visit",
            "City tour",
            "Concert ticket",
            "Theme park entry",
            "Food tasting",
            "Hiking adventure",
            "Boat cruise",
        ]
        activities_system_prompt = SystemMessage(
            content="Based on the customer profile describe the type of activities that would best suit for his trip. \n"
            "Activities list: {activity_list}."
        )
        activities_formatted_prompt = prompt_template.format(
            input=f"customer_profile: {customer_profile}, activities_list: {activity_types}",
        )
        messages = [activities_system_prompt, HumanMessage(content=activities_formatted_prompt)]
        qdrant_query_activities = llm(messages=messages)
        write_token_usage_to_csv(qdrant_query_activities)
        activities_options = query_qdrant("activities", qdrant_query_activities.content)
        all_options["activities"] = activities_options

    return all_options

def rag_response(customer_profile: str) -> SystemMessage:
    """
    Micro Agent 2 - Generate RAG Response
    Based on the customer profile, find the relevant destinations and retrive the rellevant options for flights, hotels, and activities.
    :param customer_profile: The customer profile JSON object containing the key features.
    """
    i = 0
    while i < 3:
        # Run a few times to get the rellevant data from the database, if the data is not sufficient we will try to get it again
        i += 1
        # Find the best destinations based on the customer profile
        all_options = get_database_data(customer_profile)
        # Check if we have good option from the Vector DataBase
        preformance_desired_output = "Is the data from the vector database is aligned with the customer profile and enable to build trip plans?"
        preformance_results = use_preformance_monitor(customer_profile, all_options, preformance_desired_output)
        if int(preformance_results.content[0]) == 1:
            print("The data from the vector database is sufficient enough in order to build a trip that will be aligned with the customer profile.")
            break
    
    # Use LLM to assemble full trip options
    system_prompt = SystemMessage(
        content="You are a professional travel agent. Based on the user's trip request and the available flights, hotels, and activities, build the client 3 options for a trip. Please include both the incoming and return flights (they can be from different airlines and locations), the hotels, and the activities. Make sure to include the total price for each option.",
    )
    formatted_prompt = prompt_template.format(
        input=f"customer_profile: {customer_profile}, options: {all_options}",
    )
    messages = [system_prompt, HumanMessage(content=formatted_prompt)]

    # Get the trip options from the LLM
    response = llm(messages=messages)
    write_token_usage_to_csv(response)
    return response.content