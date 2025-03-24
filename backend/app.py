import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from typing import List, Tuple, Dict
import json
import math
import pandas as pd
import numpy as np
import similarity

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'whc_sites_2021_with_ratings.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

app = Flask(__name__)
CORS(app)

def get_place_details(index):
    place = data[index]
    name = place.get("Name", "N/A")
    short_description = place.get("short_description", "N/A")
    rating = place.get("rating", "N/A")

    return {
            "Name": name,
            "Short Description": short_description,
            "Rating": rating
    }

# Sample search using json with pandas
def json_search(query):
    scores = similarity.index_search(query = query)
    ids = [id for (_, id) in scores]
    places = [get_place_details(id) for id in ids]
    def sort_key(place):
        rating = place["Rating"]
        try:
            return float(rating)
        except (ValueError, TypeError):
            return -1.0 
    sorted_places = sorted(places, key=sort_key, reverse=True)  
    return sorted_places

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
