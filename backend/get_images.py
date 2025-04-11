import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add unique id for each site, from 2019 csv file
# # Load JSON data
# with open('whc_sites_2021_with_ratings.json', 'r') as f:
#     json_data = json.load(f)

# csv_data = pd.read_csv('whc-sites-2019.csv')
# # Create a mapping from name to id_no
# name_to_id = dict(zip(csv_data['name_en'], csv_data['id_no']))

# # Update JSON data with matching id from CSV
# for site in json_data:
#     name = site.get('Name')
#     if name in name_to_id:
#         site['id'] = int(name_to_id[name])

# # Save updated JSON back to file
# with open('whc_sites_2021_with_ratings.json', 'w') as f:
#     json.dump(json_data, f, indent=2)


# Scrape images
with open('whc_sites_2021_with_ratings.json', 'r') as f:
    sites = json.load(f)

# Loop over each site and download the first gallery image
# Mutual download for missed images
for site in tqdm(sites, desc="Downloading images"):
    site_id = site.get('id')
    if not site_id:
        continue

    url = f"https://whc.unesco.org/en/list/{site_id}/gallery/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        images = soup.select("img.icaption-img.img-fluid.w-100.border")
        if not images:
            print(f"No image found for site {site_id}")
            continue

        img_url = images[0]['src']
        img_data = requests.get(img_url, timeout=10).content
        # Store the image named with unique id
        img_path = f'static/images/{site_id}.jpg'
        with open(img_path, 'wb') as f:
            f.write(img_data)

    except Exception as e:
        print(f"Failed to process site {site_id}: {e}")

