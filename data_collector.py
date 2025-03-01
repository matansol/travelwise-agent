import requests
import random
import os
import uuid


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


def process_flight_data(flights, departure_date, return_date, flight_seats_options):
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
            'dep_date': departure_date,
            'arr_date': return_date,
            'company_name': flight.get('airline', {}).get('name'),
            'seat_info': random.choice(flight_seats_options),
            'agent_commission': round(random.uniform(10, 200), 2)
        }
        results.append(flight_info)
    return results


def get_flights(from_city, departure_date, return_date, to_country=None, to_city=None, n_options=100):
    """
    Get flight options from a city to another city or country.
    """
    params = build_flight_params(from_city, to_city, to_country, n_options)
    flights = fetch_flight_data(params)
    flight_seats_options = ["Exit Row", "Window Seat", "Aisle Seat"]
    return process_flight_data(flights, departure_date, return_date, flight_seats_options)


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

def extract_destinations(user_input):
    """
    Extracts desired destination countries and cities from user input.
    :param user_input: Dictionary containing user input data.
    :return: List of countries and list of cities (converted to IATA codes).
    """
    countries = [entry['desired_country'] for entry in user_input if entry['desired_country'] != 'not specified']
    cities = [city_to_iata(entry['desired_city']) for entry in user_input if entry['desired_country'] != 'not specified']
    return countries, cities


def fetch_incoming_flights(from_city, departure_date, return_date, countries, cities):
    """
    Fetches incoming flights based on user preferences.
    :param from_city: IATA code of the departure city.
    :param departure_date: Date of departure.
    :param return_date: Date of return.
    :param countries: List of desired destination countries.
    :param cities: List of destination cities.
    :return: List of incoming flights.
    """
    incoming_flights = []
    if not countries and not cities:
        return get_flights(from_city, departure_date, return_date, n_options=10)
    for country in countries:
        for city in cities:
            incoming_flights += get_flights(from_city, departure_date, return_date, to_country=country, to_city=city, n_options=10)
    return incoming_flights


def fetch_return_flights(incoming_flights, departure_date, return_date, from_city):
    """
    Fetches return flights for identified destination cities.
    :param incoming_flights: List of incoming flights.
    :param from_city: IATA code of the departure city.
    :return: List of return flights.
    """
    return_flights = []
    cities = list(set([flight['to_city'] for flight in incoming_flights]))
    for city in cities:
        return_flights += get_flights(city, departure_date, return_date, to_city=from_city, n_options=10)
    return return_flights


def get_flights_by_user_input(user_input, from_city='JFK'):
    """
    Get flight options based on user input.
    :param user_input: Dictionary with user input data.
    :param from_city: IATA code of the city to depart from.
    :return: Incoming and return flights.
    """
    user_input = [user_input]  # Convert single dictionary to list for consistency
    countries, cities = extract_destinations(user_input)
    departure_date, return_date = user_input[0]['departure_date'], user_input[0]['return_date']
    incoming_flights = fetch_incoming_flights(from_city, departure_date, return_date, countries, cities)
    return_flights = fetch_return_flights(incoming_flights, departure_date, return_date, from_city)
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
