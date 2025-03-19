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

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'whc_sites_2021.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    """episodes_df = pd.DataFrame(data['episodes'])
    reviews_df = pd.DataFrame(data['reviews'])"""

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
"""def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json"""

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

"""
Expects [text] to be the "short description" field in the json file. Can be extended later to process the brief synthesis or other review text.
Outputs a list of tokens as a numpy array for more efficient processing later.
"""
def preprocess_description(text:str) -> np.ndarray:
    lowercase_text = text.lower()
    words_list = re.findall("[a-z]+", lowercase_text)
    return words_list
    """text = text.removeprefix("<p>").removesuffix("</p>")"""

# number of sites
n_sites = len(data)
print(" there are " + str(n_sites) + " sites")

"""
Expects: [data] to be the data from whc_sites_2021 in the json file. 
Outputs: a dict of site_id to tokenized descriptions.
"""

# create dict of doc_id to tokenized description
def create_tokenized_dict(data: List[dict]) -> dict:
    tokenized_dict = {}
    for site_id, site in enumerate(data):
      tokenize_description = preprocess_description(site['short_description'])
      tokenized_dict[site_id] = tokenize_description
    return tokenized_dict

tokenized_dict = create_tokenized_dict(data)

"""
Expects: [tokenized_dict] a dict of site_id to tokenized descriptions. 
Outputs: a dict of tuples of term to (site id, tokenized descriptions).
"""

# create inverted index (list of tuples with smaller doc_ids appearing first)
def build_inverted_index(tokenized_dict) -> dict:
    inverted_index_dict = {}

    for site_id, words in tokenized_dict.items():
      count_of_term_in_doc = {}
      
      for word in words:
        if word in count_of_term_in_doc:
          count_of_term_in_doc[word] += 1
        else:
          count_of_term_in_doc[word] = 1

      for word, count in count_of_term_in_doc.items():
        if word not in inverted_index_dict:
          inverted_index_dict[word] = []
        inverted_index_dict[word].append((site_id, count))

    for word_index in inverted_index_dict:
      inverted_index_dict[word_index].sort(key=lambda x: x[0])

    return inverted_index_dict

inv_idx = build_inverted_index(tokenized_dict)

"""
Expects: [inv_idx, n_sites] inverted index from above and number of sites. 
Outputs: a dict of terms to idf vlaues.
"""

# create dict of term to idf value
def compute_idf(inv_idx, n_sites, min_df=10, max_df_ratio=0.95):
    idf_value_dict = {}

    for word, value_list in inv_idx.items():
      precent_in = len(value_list) / n_sites

      if len(value_list) >= min_df and precent_in <= max_df_ratio:
        idf_value = math.log2(n_sites/(1 + len(value_list)))
        idf_value_dict[word] = idf_value

    return idf_value_dict

idf_dict = compute_idf(inv_idx, n_sites)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)