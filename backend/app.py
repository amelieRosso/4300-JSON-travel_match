import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
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

def get_place_details(index, filtered_data):
    place = filtered_data[index]
    name = place.get("Name", "N/A")
    short_description = place.get("short_description", "N/A")
    rating = place.get("rating", "N/A")
    inscribe_date = place.get("date_inscribed", "N/A")
    category = place.get("category_long", "N/A")
    country = place.get("Country name", "N/A")
    region = place.get("Region", "N/A")
    review_objects = place.get("reviews", [])
    reviews = [r.get("text", "") for r in review_objects if "text" in r]
    longitude = place.get("longitude", "N/A")
    latitude = place.get("latitude", "N/A")
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
             "longitude": longitude,
             "latitude": latitude,

    }

# Sample search using json with pandas
# TODO: Bert is being slow right now, so want to look into ways to have it go faster (goes faster after first search goes through, so maybe there's some caching taking place??)
def json_search(query, country_filter="", category_filter="", mode="svd", weights=None):
    filtered_data = data  # default

    if weights is None:
        weights = {}

    if country_filter:
        filtered_data = [entry for entry in filtered_data if country_filter.lower() in entry.get("Country name", "").lower()]

    if category_filter:
        filtered_data = [entry for entry in filtered_data if category_filter.lower() in entry.get("category_long", "").lower()]

    result = []

    if mode == "bert":
        scores = similarity.bert_search(query, filtered_data)

        if not scores:
            print(f"[INFO] No BERT results for query '{query}' with filters: country={country_filter}, category={category_filter}")
            return [] # If there are no BERT results, skip the process below to avoid the IndexError

        for score, local_idx in scores:
            # if local_idx >= len(filtered_data):
            #     print(f"[WARNING] Skipping out-of-range local_idx={local_idx} for filtered_indices length={len(filtered_data)}")
            #     continue # This is a temporary fix to avoid getting the IndexError. I don't love just skipping over indices > length of filtered_indices, but it's helping the error from throwing

            place = get_place_details(local_idx, filtered_data)
            # tags = similarity.extract_bert_tags(query, filtered_docs[local_idx])
            place["Similarity_Score"] = str(round(score * 100, 1))
            # place["Tags"] = tags
            place["id"] = filtered_data[local_idx]["id"]
            result.append(place)

    else:  # default: SVD
        try:
            filtered_reduced_docs, vectorizer, svd = similarity.get_reduced_docs(filtered_data)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return []
        reduced_query, _ = similarity.transform_query_to_svd(query, vectorizer, svd, weights)
        top_10 = similarity.index_search(query = query, filtered_reduced_docs = filtered_reduced_docs, vectorizer = vectorizer, svd = svd, filtered_data = filtered_data)

        for score_cos, idx, score_svd in top_10:
            place = get_place_details(idx, filtered_data)
            reduced_docs = filtered_reduced_docs[idx]
            score = (0.2*score_cos) + (0.8*score_svd) 
            tags = similarity.extract_svd_tags(reduced_query, reduced_docs, svd, vectorizer, similarity.docs[idx])
            # we need to do the actual reordering here. searching "i want a sunny place in india" gives something at the top with a lower sim score than 2nd place.
            # This actually happens with a lot of queries. Might make more sense to just order by similarity score
            # sometimes this is Nan??
            # For Nan, what if we just show no similarity score for now before debugging why
            place["Similarity_Score"] = round(score * 100, 1)
            place["Tags"] = [t[0] for t in tags]
            place["id"] = filtered_data[idx]["id"]
            result.append(place)
    # reshuffle based on score
    result = sorted(result, key = lambda place: place['Similarity_Score'], reverse=True)
    return result


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    if request.method == "POST":
        data_req = request.get_json()
        text = data_req.get("query")
        weights = data_req.get("weights", {})  # Default token weight to empty dict
        mode = data_req.get("mode", "svd")
        country_filter = data_req.get("country", "").strip().lower()
        category_filter = data_req.get("category", "").strip().lower()
    else:
        text = request.args.get("title")
        weights_json = request.args.get("weights", "{}")
        weights = json.loads(weights_json)
        print(type(text))
        print(text)
        country_filter = request.args.get("country", "").strip().lower()
        category_filter = request.args.get("category", "").strip().lower()
        mode = request.args.get("mode", "svd") 
    return json_search(text, country_filter, category_filter, mode, weights)

"""@app.route("/filters")
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
    })"""

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
