import requests
import random
import os
import uuid
from sentence_transformers import SentenceTransformer


def build_flight_params(from_city, to_city, to_country, n_options):
    """
    Builds query parameters for the API request.
    """
    FLIGHTS_API_KEY = os.getenv("AVIATION_API_KEY")
    params = {'access_key': FLIGHTS_API_KEY, 'dep_iata': from_city, 'limit': n_options}
    if to_city:
        params['arr_iata'] = to_city
    if to_country:
        params['arr_country'] = to_country
    return params


def fetch_flight_data(params):
    """
    Sends a request to the API and retrieves flight data.
    """
    base_url = 'https://api.aviationstack.com/v1/flights'
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print(f"Error: {response}")
        print(response.text)
        return []


def process_flight_data(flights, flight_seats_options, embedding_model):
    """
    Processes raw flight data and formats it into a structured list.
    """
    results = []
    for flight in flights:
        flight_info = {
            'id': flight.get('flight', {}).get('iata'),
            'price_dollars': round(random.uniform(100, 1000), 2),
            'from_city': flight.get('departure', {}).get('iata'),
            'from_airport': flight.get('departure', {}).get('airport'),
            'to_airport': flight.get('arrival', {}).get('airport'),
            'to_city': flight.get('arrival', {}).get('iata'),
            'company_name': flight.get('airline', {}).get('name'),
            'seat_info': random.choice(flight_seats_options),
            'agent_commission': round(random.uniform(10, 200), 2)
        }
        flight_str = (f"{flight_info['id']} from {flight_info['from_city']}, airport {flight_info['from_airport']} "
                      f"to {flight_info['to_city']}, airport {flight_info['to_airport']}, price: ${flight_info['price_dollars']}, "
                      f"company: {flight_info['company_name']}, seat: {flight_info['seat_info']}, "
                      f"commission: ${flight_info['agent_commission']}")
        results.append(embedding_model.encode(flight_str))
    return results


def get_flights(from_city, embedding_model, to_country=None, to_city=None, n_options=1):
    """
    Get flight options from a city to another city or country.
    :param from_city: IATA code of the departure city.
    :param embedding_model: SentenceTransformer model for encoding flight descriptions.
    :param to_country: IATA code of the destination country.
    :param to_city: IATA code of the destination city.

    :return: List of flight options.
    """
    params = build_flight_params(from_city, to_city, to_country, n_options)
    flights = fetch_flight_data(params)
    flight_seats_options = ["Exit Row", "Window Seat", "Aisle Seat"]
    return process_flight_data(flights, flight_seats_options, embedding_model)


def city_to_iata(city_name):
    """
    Get the IATA code of a city.
    :param city_name: Name of the city
    """
    city_iata_map = {
        'New York': 'JFK',
        'Los Angeles': 'LAX',
        'Chicago': 'ORD',
        'Houston': 'IAH',
        'Phoenix': 'PHX',
        'Philadelphia': 'PHL',
        'San Antonio': 'SAT',
        'San Diego': 'SAN',
        'Dallas': 'DFW',
        'San Jose': 'SJC',
        'Austin': 'AUS',
        'Jacksonville': 'JAX',
        'Fort Worth': 'DFW',
        'Columbus': 'CMH',
        'Indianapolis': 'IND',
        'Charlotte': 'CLT',
        'Seattle': 'SEA',
        'Denver': 'DEN',
        'Washington DC': 'DCA',
        'Boston': 'BOS',
        'Nashville': 'BNA',
        'Detroit': 'DTW',
        'Las Vegas': 'LAS',
        'Memphis': 'MEM',
        'Portland': 'PDX',
        'Oklahoma City': 'OKC',
        'Louisville': 'SDF',
        'Baltimore': 'BWI'
    }

    return city_iata_map.get(city_name, None)

# def extract_destinations(user_input):
#     """
#     Extracts desired destination countries and cities from user input.
#     :param user_input: Dictionary containing user input data.
#     :return: List of countries and list of cities (converted to IATA codes).
#     """
#     countries = [entry['desired_country'] for entry in user_input if entry['desired_country'] != 'not specified']
#     cities = [city_to_iata(entry['desired_city']) for entry in user_input if entry['desired_country'] != 'not specified']
#     return countries, cities


def fetch_incoming_flights(from_city, cities, embedding_model):
    """
    Fetches incoming flights based on user preferences.
    :param from_city: IATA code of the departure city.
    :param cities: List of destination cities.
    :param embedding_model: SentenceTransformer model for encoding flight descriptions.
    :return: List of incoming flights.
    """
    incoming_flights = []
    for city in cities:
        incoming_flights += get_flights(from_city, embedding_model, to_city=city, n_options=1)
    return incoming_flights


def fetch_return_flights(incoming_flights, from_city, embedding_model):
    """
    Fetches return flights for identified destination cities.
    :param incoming_flights: List of incoming flights.
    :param from_city: IATA code of the departure city.
    :param embedding_model: SentenceTransformer model for encoding flight descriptions.
    :return: List of return flights.
    """
    return_flights = []
    cities = list(set([flight['to_city'] for flight in incoming_flights]))
    for city in cities:
        return_flights += get_flights(city, embedding_model, to_city=from_city, n_options=1)
    return return_flights


def get_flights_by_params(cities,  embedding_model, from_city='JFK'):
    """
    Get flight options based on user input.
    :param cities: List of destination cities.
    :param from_city: IATA code of the city to depart from.
    :param embedding_model: SentenceTransformer model for encoding flight descriptions.
    :return: Incoming and return flights.
    """
    incoming_flights = fetch_incoming_flights(from_city, cities, embedding_model)
    return_flights = fetch_return_flights(incoming_flights, from_city, embedding_model)
    return incoming_flights, return_flights


def generate_hotel_data(num_records, city, embedding_model):
    """
    Generates a list of hotel data with randomized details, pricing, and embeddings.
    :param num_records: Number of hotel records to generate.
    :param city: The city in which the hotels are located.
    :param embedding_model: SentenceTransformer model for encoding hotel descriptions.
    :return: List of dictionaries containing hotel data.
    """
    hotel_names = ["HotelComfort", "StayEasy", "LuxuryLodge", "BudgetInn", "FourSeasons", "GrandPlaza", "SkylineSuites",
                   "RoyalPalace", "UrbanRetreat", "EliteResidency"]
    room_types = ["Single", "Double", "Suite", "Deluxe", "Penthouse"]
    hotels = []

    for _ in range(num_records):
        hotel = {
            "id": str(uuid.uuid4()),
            "hotel_name": random.choice(hotel_names),
            "city": city,
            "hotel_rate": round(random.uniform(3.0, 5.0), 1),
            "room_info": random.choice(room_types),
            "room_size (m^2)": round(random.uniform(20, 200), 2),
            "price($)": round(random.uniform(50, 500), 2),
            "commission": round(random.uniform(5, 100), 2),
        }
        hotel_str = (f"{hotel['hotel_name']} in {hotel['city']} - room: {hotel['room_info']}, "
                     f"room_size: {hotel['room_size (m^2)']}m^2, price: ${hotel['price($)']}, "
                     f"commission: ${hotel['commission']}")
        hotel["vector"] = embedding_model.encode(hotel_str)
        hotels.append(hotel)
    return hotels


def generate_activity_data(num_records, location, embedding_model):
    """
    Generates a list of activity data with randomized details, pricing, and embeddings.
    :param num_records: Number of activity records to generate.
    :param location: The location where the activities are available.
    :param embedding_model: SentenceTransformer model for encoding activity descriptions.
    :return: List of dictionaries containing activity data.
    """
    activity_types = ["Museum visit", "City tour", "Concert ticket", "Theme park entry", "Food tasting",
                      "Hiking adventure", "Boat cruise"]
    company_names = ["FunTimes", "AdventureX", "CityTours", "ExperienceIt", "ExploreMore", "ThrillSeekers"]
    activities = []

    for _ in range(num_records):
        activity = {
            "id": str(uuid.uuid4()),
            "activity_info": random.choice(activity_types),
            "location": location,
            "company_name": random.choice(company_names),
            "company_rate": round(random.uniform(1.0, 5.0), 1),
            "price($)": round(random.uniform(20, 300), 2),
            "commission": round(random.uniform(5, 50), 2),
        }
        activity_str = (f"{activity['activity_info']} in {activity['location']} by {activity['company_name']} - "
                        f"company_rate: {activity['company_rate']}, price: ${activity['price($)']}, "
                        f"commission: ${activity['commission']}")
        activity["vector"] = embedding_model.encode(activity_str)
        activities.append(activity)
    return activities

def create_data_lists(embedding_model):
    """
    Create a list of hotel data for a given city.
    :param embedding_model: SentenceTransformer model for encoding hotel descriptions.
    :return: List of hotel data.
    """
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']

    for city in cities:
        hotels = generate_hotel_data(10, city, embedding_model)
        activities = generate_activity_data(10, city, embedding_model)
        flights = get_flights_by_params(cities, embedding_model, from_city=city)
    return hotels, activities, flights
