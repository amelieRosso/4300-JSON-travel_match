import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from typing import List, Tuple, Dict
import json
import math
#import pandas as pd
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
    inscribe_date = place.get("date_inscribed", "N/A")
    category = place.get("category_long", "N/A")
    country = place.get("Country name", "N/A")
    region = place.get("Region", "N/A")
    review_objects = place.get("reviews", [])
    reviews = [r.get("text", "") for r in review_objects if "text" in r]



    return {
            "Name": name,
            "Short Description": short_description,
            "Rating": rating,
            "Inscribe Date": inscribe_date,
            "Category": category,
            "Country": country,
            "Region": region,
            "Reviews": reviews
    }

# Sample search using json with pandas
def json_search(query):
    reduced_query, _ = similarity.transform_query_to_svd(query)
    scores = similarity.svd_index_search(reduced_query=reduced_query, reduced_docs= similarity.reduced_docs)
    top_10 = scores[:10]
    result = []
    for score, idx in top_10:
        place = get_place_details(idx)
        reduced_docs = similarity.reduced_docs[idx]
        tags = similarity.extract_svd_tags(reduced_query, reduced_docs, similarity.svd, similarity.vectorizer)
        place["Similarity_Score"]= str(round(score*100,1))+"%"
        place["Tags"] = tags
        print(place['Name'])
        place["id"] = data[idx]["id"]
        result.append(place)

    return result


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    print(text)
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
