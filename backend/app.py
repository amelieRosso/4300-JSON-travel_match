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
def json_search(query, country_filter="", category_filter="", mode="svd"):
    filtered_docs = []
    filtered_indices = []

    for i, entry in enumerate(data):
        country = entry.get("Country name", "").lower()
        category = entry.get("category_long", "").lower()

        if (not country_filter or country_filter.lower() in country) and (not category_filter or category_filter.lower() in category):
            full_text = entry.get("Name", "") + " " + entry.get("short_description", "")
            reviews = entry.get("reviews", [])
            full_text += " " + " ".join(r.get("text", "") for r in reviews)
            filtered_docs.append(full_text)
            filtered_indices.append(i)

    if not filtered_indices:
        return []

    result = []

    if mode == "bert":
        scores = similarity.bert_search(query, filtered_docs)

        for score, local_idx in scores:
            global_idx = filtered_indices[local_idx]
            place = get_place_details(global_idx)
            tags = similarity.extract_bert_tags(query, similarity.docs[global_idx])
            place["Similarity_Score"] = str(round(score * 100, 1)) + "%"
            place["Tags"] = tags
            place["id"] = data[global_idx]["id"]
            result.append(place)
          

    else:  # default: SVD
        reduced_query, _ = similarity.transform_query_to_svd(query)
        top_10 = similarity.index_search(query, subset_indices=set(filtered_indices))

        for score_cos, idx, score_svd in top_10:
            place = get_place_details(idx)
            reduced_docs = similarity.reduced_docs[idx]
            tags = similarity.extract_svd_tags(reduced_query, reduced_docs, similarity.svd, similarity.vectorizer, similarity.docs[idx])
            score = (0.2*score_cos) + (0.8*score_svd) 
            # we need to do the actual reordering here. searching "i want a sunny place in india" gives something at the top with a lower sim score than 2nd place.
            # sometimes this is Nan??
            place["Similarity_Score"] = round(score * 100, 1)
            place["Tags"] = tags
            place["id"] = data[idx]["id"]
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
    mode = request.args.get("mode", "svd") 
    return json_search(text, country_filter, category_filter, mode)

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
