from src.micro_agents import *


def get_database_data(customer_profile: str) -> str:
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


    
    # Query Qdrant collections for flights, hotels, and activities
    flights_options = query_qdrant("flights", qdrant_query_destination.content)
    hotels_options = query_qdrant("hotels", qdrant_query_hotels.content)
    activities_options = query_qdrant("activities", qdrant_query_activities.content)

    # Combine all options into a single list
    all_options = {"flights": flights_options, "hotels": hotels_options, "activities": activities_options}
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
        preformance_desired_output = "Is the data from the vector database is sufficient enough in order to build a trip that will be aligned with the customer profile?"
        preformance_results = use_preformance_monitor(customer_profile, all_options, preformance_desired_output)
        if int(preformance_results.content[0]) == 1:
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