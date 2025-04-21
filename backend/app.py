import json
import os
import re
from flask import Flask, render_template, request, jsonify
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

# Load the country code map
code_map_path = os.path.join(current_directory, 'country_code.json')
with open(code_map_path, 'r', encoding='utf-8') as f:
    country_code = json.load(f)

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
    # code = country_code.get(country, "unknown")
    # print(f"Looking up code for: {country} -> {country_code.get(country)}")
    iso_codes_raw = place.get("iso_code", "")
    iso_codes = [code.strip() for code in iso_codes_raw.split(",") if code.strip()]

    return {
            "Name": name,
            "Short Description": short_description,
            "Rating": rating,
            "Inscribe Date": inscribe_date,
            "Category": category,
            "Country": country,
            "Region": region,
            "Reviews": reviews,
            # "Country_Code": code
            "ISO_Codes": iso_codes,

    }

# Sample search using json with pandas
def json_search(query, country_filter = "", category_filter = ""):
    filtered_docs = []
    filtered_indices = []

    for i, entry in enumerate(data):
        country = entry.get("Country name", "").lower()
        category = entry.get("category_long", "").lower()

        if(not country_filter or country_filter in country) and (not category_filter or category_filter in category):
            filtered_docs.append(entry)
            filtered_indices.append(i)
    
    if not filtered_indices:
        return []
    
    reduced_query, _ = similarity.transform_query_to_svd(query)
    #scores = similarity.svd_index_search(reduced_query=reduced_query, reduced_docs= similarity.reduced_docs)
    #top_10 = scores[:10]
    top_10 = similarity.index_search(query, subset_indices = set(filtered_indices))

    #print(f"top 10 {top_10}")

    result = []
    for score_cos, idx, score_svd in top_10:
        place = get_place_details(idx)
        reduced_docs = similarity.reduced_docs[idx]
        tags = similarity.extract_svd_tags(reduced_query, reduced_docs, similarity.svd, similarity.vectorizer)
        score = (score_cos + score_svd) / 2
        place["Similarity_Score"]= str(round(score*100,1))+"%"
        place["Tags"] = tags
        #print(place['Name'])
        place["id"] = data[idx]["id"]
        # place["Country_Code"] = country_code.get(place["Country"], "unknown")
        result.append(place)

    return result


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    print(type(text))
    print(text)
    country_filter = request.args.get("country", "").strip().lower()
    category_filter = request.args.get("category", "").strip().lower()
    return json_search(text, country_filter, category_filter)

@app.route("/filters")
def filters():
    with open("whc_sites_2021_with_ratings.json") as f:
        data = json.load(f)

    country_tokens = set()
    category_tokens = set()

    for entry in data:
        raw_countries = entry.get("Country name", "")
        split_countries = [c.strip() for c in raw_countries.split(",") if c.strip()]
        country_tokens.update(split_countries)

    categories = sorted(set(entry["category_long"] for entry in data if entry.get("category_long")))

    return jsonify({
        "countries": sorted(country_tokens),
        "categories": categories
    })

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
