
"""
extract_main_table.py
Fetch the main Elo table and save as CSV, without pd.read_html warnings.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from urllib.parse import urljoin

INDEX_URL = "https://tennisabstract.com/reports/atp_elo_ratings.html"
OUTPUT_CSV = "atp_elo_directory.csv"

def fetch_and_save_directory():
    # 1. Download page
    print(f"Fetching Elo data from {INDEX_URL}...")
    resp = requests.get(INDEX_URL, timeout=30)
    resp.raise_for_status()

    # 2. Parse the Elo table
    print("Parsing HTML table...")
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="reportable")
    if not table:
        raise RuntimeError("Could not find table#reportable")

    # 3. Read into DataFrame via StringIO
    html = StringIO(str(table))
    df = pd.read_html(html, header=0)[0]

    # 4. Extract profile URLs
    hrefs = []
    for row in table.tbody.find_all("tr"):
        a = row.find("a", href=True)
        hrefs.append(a["href"] if a else "")

    # 5. Use urljoin for robustly creating absolute URLs from relative links
    df["ProfileURL"] = [urljoin(INDEX_URL, h) for h in hrefs]

    # 6. Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} players to {OUTPUT_CSV}")

if __name__ == "__main__":
    fetch_and_save_directory()
