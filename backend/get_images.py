import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from lxml import etree
import os

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


# Updated the unique id for the newest web xml file
# If the id for site doesn't exist in json, scrap id from whc_unesco_list file
with open('whc_sites_2021_with_ratings.json', 'r') as f:
    sites = json.load(f)

tree = etree.parse('whc_unesco_list.xml')
root = tree.getroot()
items = root.xpath('//item')

title_to_id = {}
for item in items:
    title = item.findtext('title')
    link = item.findtext('link')
    if title and link and '/list/' in link:
        # Get the id from xml
        site_id = link.split('/')[-1]
        title_to_id[title.strip()] = int(site_id)

# Add missing ids to the JSON
missing_names = []
for site in sites:
    if 'id' not in site or site['id'] is None:
        name = site.get('Name', '').strip()
        if name in title_to_id:
            site['id'] = title_to_id[name]
        else:
            missing_names.append(name)
print(f"[INFO] Still missing id for {len(missing_names)} sites.")
if missing_names:
    print("Missing site names:")
    for name in missing_names:
        print(f" - {name}")

with open('whc_sites_2021_with_ratings.json', 'w', encoding='utf-8') as f:
    json.dump(sites, f, indent=2)

# Loop over each site and download the first gallery image
# Mutual download for missed images
skipped = 0
for site in tqdm(sites, desc="Downloading images"):
    site_id = site.get('id')
    if not site_id:
        continue
    
    # Only scrape missed images
    img_path = f'static/images/{site_id}.jpg'
    if os.path.exists(img_path):
        skipped += 1
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

print(f"[DONE] Skipped {skipped} already-downloaded images.")