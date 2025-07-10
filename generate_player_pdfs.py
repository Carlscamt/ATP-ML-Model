import pandas as pd
import asyncio
import os
from unidecode import unidecode
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# --- Configuration ---
SOFASCORE_CSV = "sofascore_filtered_matches.csv"
ELO_DIRECTORY_CSV = "atp_elo_directory.csv"
OUTPUT_DIR = "player_profiles_pdf"

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

async def fetch_and_save_pdfs(urls_to_fetch):
    """Uses crawl4ai to fetch and save PDFs for a given set of URLs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")

    # Redirect crawl4ai logger to a file to prevent console encoding errors
    import logging
    logging.basicConfig(filename='crawler.log', level=logging.INFO, 
                        format='%(asctime)s - %(message)s')

    async with AsyncWebCrawler() as crawler:
        for player_name, url in urls_to_fetch.items():
            print(f"Fetching PDF for: {player_name} from {url}")
            try:
                run_config = CrawlerRunConfig(pdf=True, page_timeout=45000) # 45s timeout
                result = await crawler.arun(url, config=run_config)

                if result.success and result.pdf:
                    # Sanitize player name for a valid filename
                    safe_filename = unidecode(player_name).replace(' ', '_').replace('/', '_') + ".pdf"
                    filepath = os.path.join(OUTPUT_DIR, safe_filename)

                    with open(filepath, "wb") as f:
                        f.write(result.pdf)
                    print(f"  ✅ Successfully saved PDF to {filepath}")
                else:
                    print(f"  ❌ Failed to generate PDF for {player_name}. Error: {result.error_message}")

            except Exception as e:
                print(f"  ❌ An unexpected error occurred for {player_name}: {e}")

async def main():
    """Main orchestration function."""
    print("--- Starting Player Profile PDF Generation ---")

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

    # --- 3. Fetch and Save PDFs ---
    if urls_to_fetch:
        await fetch_and_save_pdfs(urls_to_fetch)
    else:
        print("No URLs to fetch. Exiting.")

    print("\n--- PDF Generation Complete ---")

if __name__ == "__main__":
    # Ensure playwright browsers are installed for crawl4ai
    # You can run `playwright install` in your terminal
    asyncio.run(main())