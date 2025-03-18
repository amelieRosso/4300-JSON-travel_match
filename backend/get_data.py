import json
import requests
import time
import unicodedata
import os

# Set Google Places API key 
API_KEY = "PERSONAL KEY"

# Clean and normalize query
def clean_query(text):
    text = unicodedata.normalize('NFKD', text)
    return " ".join(text.split())

# Get rating and add them to json
def get_rating_from_api(input_json_path, output_json_path):
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            sites = json.load(f)
    except Exception as e:
        print(f"Error loading input JSON: {e}")
        return False

    updated_count = 0
    total_sites = len(sites)

    for i, site in enumerate(sites):
        name = site.get("Name", "")
        country = site.get("Country name", "")

        # Try multiple query formats to improve match rate
        query_options = [
            clean_query(f"{name}, {country}"),
            clean_query(f"{name} tourist attraction in {country}"),
            clean_query(f"{name} UNESCO site {country}")
        ]

        matched = False

        for query in query_options:
            print(f"[{i+1}/{total_sites}] Trying query: {query}")

            # Get place_id via Text Search
            text_search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            text_search_params = {
                "query": query,
                "key": API_KEY
            }

            try:
                text_response = requests.get(text_search_url, params=text_search_params)
                text_data = text_response.json()

                # Print response structure for debug
                print(json.dumps(text_data, indent=2))

                if text_data.get("status") == "OK" and text_data.get("results"):
                    place_id = text_data["results"][0].get("place_id")
                    site["place_id"] = place_id  # Add place_id to JSON

                    # Use Place Details API when there is avaliable query result
                    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                    details_params = {
                        "place_id": place_id,
                        "fields": "rating,formatted_address",
                        "key": API_KEY
                    }

                    details_response = requests.get(details_url, params=details_params)
                    details_data = details_response.json()

                    # Once successfully query, jump to next site query
                    if details_data.get("status") == "OK":
                        result = details_data.get("result", {})
                        site["rating"] = result.get("rating")
                        site["formatted_address"] = result.get("formatted_address")

                        updated_count += 1
                        matched = True
                        break 
                    else:
                        print(f"Details API failed for {query}: {details_data.get('status')}")
                        site["rating"] = None
                else:
                    print(f"No results for query: {query}")

            except Exception as e:
                print(f"Error for {query}: {e}")
                site["rating"] = None

            time.sleep(1.5)

        if not matched:
            site["rating"] = None
            site["formatted_address"] = None
            site["place_id"] = None

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(sites, f, ensure_ascii=False, indent=2)
        print(f"Successfully updated {updated_count} out of {total_sites} sites")
        return True
    except Exception as e:
        print(f"Error saving output JSON: {e}")
        return False