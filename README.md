# TravelWise Agent

TravelWise Agent is an AI-driven assistant designed to support travel agents in managing their tasks efficiently. 
This project leverages advanced language models and data processing tools to streamline operations within the travel industry.
Please note that currently, the agent can plan travels from and to the following cities: New York, Los Angeles, Chicago, Houston, Phoenix.
Additionally, in order to use the data collector file to collect the flights' data, AVIATION's API key is required. 

## Project Overview

The aim of this project is to develop an AI agent that assists travel agents by automating data collection, classification, and analysis tasks. 
The system utilizes machine learning models to process travel-related data, providing valuable insights and recommendations.

## Features

- **Automated Data Collection**: Gathers and processes travel-related data to assist agents in making informed decisions.
- **Intelligent Classification**: Utilizes LLMs to categorize and prioritize tasks effectively.

## Installation

To set up the TravelWise Agent environment, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/matansol/travelwise-agent.git
   cd travelwise-agent
   ```


2. **Create and activate a conda environment**:

   ```bash
   conda env create -f environment.yml
   conda activate travelwise-agent
   ```


3. **Install additional dependencies**:

   ```bash
   pip install -r requirements.txt
   ```


## Usage

- **Data Collection**: Run `data_collector.py` to fetch and process travel data.
  Important note: The data returned by this file is primarily synthetic and is intended to demonstrate the proof of concept.
- **Main Application**: Execute `main.py` to start the TravelWise Agent interface.
- **Notebooks**: Explore `agents.ipynb`.

## Contributors

- [matansol](https://github.com/matansol)
- [cohen-or-github](https://github.com/cohen-or-github)
- [saharad1](https://github.com/saharad1)
