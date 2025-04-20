import requests
from bs4 import BeautifulSoup
import json

# Website lists all country involved
url = "https://whc.unesco.org/en/statesparties/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
country_code_map = {}

# Target the table with id="t1" and loop over each row
table = soup.find("table", {"id": "t1"})
if not table:
  print("Table not found.")
else:
  rows = table.find_all("tr")[1:]  # Skip the header row

  for row in rows:
    cells = row.find_all("td")
    if len(cells) >= 2:
      country_td = cells[1]
      country_name = country_td.get("sorttable_customkey", "").strip()
      link_tag = country_td.find("a")
      if link_tag and "href" in link_tag.attrs:
        code = link_tag["href"].split("/")[-1]
        country_code_map[country_name] = code

# Print to check
print(f"Parsed {len(country_code_map)} countries")
for name, code in sorted(country_code_map.items()):
  print(f'"{name}": "{code}",')

with open("country_code.json", "w", encoding="utf-8") as f:
  json.dump(country_code_map, f, indent=2)
