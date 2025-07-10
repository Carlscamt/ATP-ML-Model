import pandas as pd
import asyncio
import os
from unidecode import unidecode
from playwright.async_api import async_playwright
from io import StringIO
import json # Import json module
import time # Import time module for delays
from datetime import date

# --- Configuration ---
SOFASCORE_CSV = "sofascore_filtered_matches.csv"
ELO_DIRECTORY_CSV = "atp_elo_directory.csv"
OUTPUT_DIR = "C:/Users/Carlos/Documents/ATP_ML_Refactored/source_codes/player_profiles_data" # Changed output directory name

# --- Force UTF-8 output for console ---
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def create_player_url_mapping(elo_df):
    """Creates a dictionary mapping player last names to their profile URLs."""
    player_map = {}
    for _, row in elo_df.iterrows():
        player_name = row['Player']
        # Normalize the full name for better matching (e.g., handle accents)
        normalized_name = unidecode(player_name).lower()
        player_map[normalized_name] = row['ProfileURL']
    return player_map

def find_url_for_player(player_name, player_map):
    """Finds a profile URL for a player by matching their full name or last name."""
    # Normalize the SofaScore player name
    normalized_player_name = unidecode(player_name).lower()

    # 1. Try to match the full name directly
    if normalized_player_name in player_map:
        return player_map[normalized_player_name]

    # 2. If no full name match, try matching by last name
    # This helps with variations like "Novak Djokovic" vs "Djokovic N."
    player_last_name = normalized_player_name.split(' ')[-1]
    for map_name, url in player_map.items():
        map_last_name = map_name.split(' ')[-1]
        if player_last_name == map_last_name:
            return url # Return the first match found

    return None

def calculate_age(dob_yyyymmdd):
    """Calculates age from a YYYYMMDD date string."""
    if not dob_yyyymmdd:
        return None
    try:
        dob_str = str(dob_yyyymmdd)
        birth_year = int(dob_str[0:4])
        birth_month = int(dob_str[4:6])
        birth_day = int(dob_str[6:8])
        
        today = date.today()
        age = today.year - birth_year - ((today.month, today.day) < (birth_month, birth_day))
        return age
    except ValueError:
        return None

async def fetch_and_save_tables(urls_to_fetch):
    """Uses Playwright to fetch HTML and extract tables and player bio for a given set of URLs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")

    # Redirect logging to a file to prevent console encoding errors
    import logging
    logging.basicConfig(filename='player_tables_scraper.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True) # Run headless for automation
        
        for player_name, url in urls_to_fetch.items():
            print(f"Processing tables for: {player_name} from {url}")
            page = None # Initialize page to None
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until='domcontentloaded', timeout=60000) # Wait for DOM to be loaded
                
                # Add a delay to avoid rate limiting
                time.sleep(5)

                # Wait for a common table selector to appear, with a timeout
                try:
                    await page.wait_for_load_state('networkidle', timeout=60000) # Wait for network to be idle for up to 60 seconds
                except Exception as e:
                    print(f"  ⚠️ Page did not reach network idle state for {player_name} within timeout: {e}")
                    # Continue to get content even if no table is found, to allow for manual inspection

                html_content = await page.content() # Get the fully rendered HTML

                # Check for Cloudflare access denied
                if "Access denied" in html_content or "You are being rate limited" in html_content:
                    print(f"  ❌ Access denied by Cloudflare for {player_name}. Skipping this player.")
                    logging.error(f"Access denied by Cloudflare for {player_name} ({url}).")
                    continue # Skip to the next player

                # Save HTML content to a temporary file for inspection
                temp_html_path = os.path.join(OUTPUT_DIR, f"temp_{unidecode(player_name).replace(' ', '_')}.html")
                with open(temp_html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"  HTML content saved to {temp_html_path} for inspection.")

                # Sanitize player name for a valid directory name
                safe_player_dir = unidecode(player_name).replace(' ', '_').replace('/', '_')
                player_output_dir = os.path.join(OUTPUT_DIR, safe_player_dir)
                os.makedirs(player_output_dir, exist_ok=True)

                # --- Extract Player Bio Information ---
                player_bio = {}
                try:
                    # Directly access JavaScript variables
                    dob_js = await page.evaluate("window.dob")
                    hand_js = await page.evaluate("window.hand")
                    currentrank_js = await page.evaluate("window.currentrank")

                    print(f"  DEBUG: dob_js = {dob_js}, hand_js = {hand_js}, currentrank_js = {currentrank_js}")

                    if dob_js:
                        player_bio['Age'] = calculate_age(dob_js)

                    if hand_js:
                        player_bio['Hand'] = "Right-handed" if hand_js == 'R' else "Left-handed"

                    if currentrank_js:
                        player_bio['ATP Rank'] = int(currentrank_js)

                    if player_bio:
                        bio_filepath = os.path.join(player_output_dir, "player_bio.json")
                        with open(bio_filepath, "w", encoding="utf-8") as f:
                            json.dump(player_bio, f, indent=4)
                        print(f"    ✅ Saved player bio to {bio_filepath}")
                except Exception as e:
                    print(f"  ❌ Error extracting player bio for {player_name}: {e}")

                table_ids = [
                    'recent-results', 'career-splits', 'head-to-heads', 'tour-results',
                    'match-results', 'year-end-rankings', 'titles-finals', 'recent-events',
                    'recent-finals', 'tour-years', 'chall-years', 'news-analysis',
                    'mcp-serve', 'mcp-return', 'mcp-rally', 'mcp-tactics',
                    'career-splits-chall', 'last52-splits-chall', 'doubles', 'mixed-doubles',
                    'pbp-points', 'pbp-games', 'pbp-stats', 'serve-speed', 'winners-errors'
                ]
                
                extracted_any_table = False
                for table_id in table_ids:
                    try:
                        # Use page.evaluate to get the outerHTML of the specific table
                        table_html = await page.evaluate(f"document.getElementById('{table_id}')?.outerHTML")
                        
                        if table_html:
                            tables = pd.read_html(StringIO(table_html))
                            if tables:
                                for i, table_df in enumerate(tables):
                                    table_filepath = os.path.join(player_output_dir, f"{table_id}_{i+1}.csv")
                                    table_df.to_csv(table_filepath, index=False, encoding='utf-8')
                                    print(f"    ✅ Saved table '{table_id}' to {table_filepath}")
                                    extracted_any_table = True
                            else:
                                print(f"  No data found in table '{table_id}' for {player_name}.")
                        else:
                            print(f"  Table with ID '{table_id}' not found on page for {player_name}.")
                    except Exception as e:
                        print(f"  ❌ Error extracting table '{table_id}' for {player_name}: {e}")
                
                if not extracted_any_table:
                    print(f"  No tables were successfully extracted for {player_name}.")

            except Exception as e:
                print(f"  ❌ An unexpected error occurred for {player_name}: {e}")
            finally:
                if page: # Close the page if it was opened
                    await page.close()
        await browser.close() # Close the browser after all players are processed

async def main():
    """Main orchestration function."""
    print("--- Starting Player Profile Table Extraction ---")

    # --- 1. Load Data ---
    try:
        sofascore_df = pd.read_csv(SOFASCORE_CSV)
        elo_directory_df = pd.read_csv(ELO_DIRECTORY_CSV)
        print(f"Loaded {len(sofascore_df)} matches from {SOFASCORE_CSV}")
        print(f"Loaded {len(elo_directory_df)} players from {ELO_DIRECTORY_CSV}")
    except FileNotFoundError as e:
        print(f"Error: Missing required file - {e.filename}. Please run the prerequisite scripts.")
        return

    # --- 2. Prepare Player & URL Data ---
    player_url_map = create_player_url_mapping(elo_directory_df)

    # Get a unique list of all players from the sofascore file
    p1_names = sofascore_df['Player 1'].dropna().unique()
    p2_names = sofascore_df['Player 2'].dropna().unique()
    all_unique_players = set(p1_names) | set(p2_names)
    print(f"Found {len(all_unique_players)} unique players to process.")

    urls_to_fetch = {}
    for player_name in all_unique_players:
        url = find_url_for_player(player_name, player_url_map)
        if url:
            urls_to_fetch[player_name] = url
        else:
            print(f"  - Could not find a URL for player: {player_name}")

    print(f"Found profile URLs for {len(urls_to_fetch)} players.")

    # --- 3. Fetch and Save Tables ---
    if urls_to_fetch:
        await fetch_and_save_tables(urls_to_fetch)
    else:
        print("No URLs to fetch. Exiting.")

    print("--- Table Extraction Complete ---")

if __name__ == "__main__":
    # Ensure playwright browsers are installed
    # You can run `playwright install` in your terminal
    asyncio.run(main())