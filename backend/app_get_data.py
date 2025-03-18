import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from get_data import get_rating_from_api


# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the actual input JSON file relative to the current script
input_json_path = os.path.join(current_directory, 'whc_sites_2021.json')

# Specify the path to the actual output JSON file relative to the current script
output_json_path = os.path.join(current_directory, 'whc_sites_2021_with_ratings.json')

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    try:
        # Load the input JSON file
        with open(input_json_path, 'r', encoding='utf-8') as f:
            sites = json.load(f)
        
        # Get ratings from API and update the JSON
        get_rating_from_api(input_json_path, output_json_path)
        
        # Load the updated JSON to display on the home page
        with open(output_json_path, 'r', encoding='utf-8') as f:
            updated_sites = json.load(f)
        
        # Ignore
        return render_template('index.html', sites=updated_sites, title="UNESCO World Heritage Sites")
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)