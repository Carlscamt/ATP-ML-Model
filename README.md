# ATP Match Prediction Model Suite

This repository contains a suite of Python scripts designed to gather data, train a tennis match prediction model, and make predictions on upcoming ATP matches.

## The Workflow

The scripts are designed to be run in a sequence:

1. **`event_scraper.py`**: Scrapes upcoming matches to know which players to look up.
2. **`elo_fetcher.py`**: Creates a master list of all players and their profile URLs.
3. **`generate_player_tables.py`**: Fetches detailed stats for the specific players who are playing soon.
4. **`model_training.py`**: Trains the prediction model on historical data.
5. **`interactive_predictor.py`**: Uses the trained model to make predictions.

## Scripts

### 1. Model Training (`model_training.py`)

**Purpose:**
This is the core script for training the XGBoost-based tennis match prediction model. It handles everything from data acquisition to model evaluation and serialization.

**How to Run:**
```bash
python model_training.py
```

**Functionality:**
- **Data Acquisition**: Downloads historical ATP match data (2010-2024) from Jeff Sackmann's public GitHub repository.
- **Feature Engineering**: Processes the raw data to create a rich feature set, including ELO/Glicko-2 ratings, surface-specific ELO, head-to-head history, player form, activity metrics, physical attributes, and career stats.
- **Training**: Automatically detects and utilizes a GPU for faster training if available (nvidia-smi), otherwise falls back to CPU.
- **Evaluation**: Reports the model's accuracy and AUC score on a validation set.

**Output:**
- `tennis_model_gpu_complete_{timestamp}.json`: The trained XGBoost model.
- `surface_encoder_complete_{timestamp}.pkl`: The scikit-learn LabelEncoder for court surfaces.
- `feature_importance_complete_{timestamp}.csv`: A report of the most influential features.
- `tennis_feature_frame.pkl`: The complete, processed data frame used for feature creation.

### 2. Interactive Predictor (`interactive_predictor.py`)

**Purpose:**
Provides a command-line interface to make predictions on a single, user-defined match using a pre-trained model.

**How to Run:**
```bash
python interactive_predictor.py
```

**Functionality:**
- **Model Loading**: Automatically loads the most recently trained model (.json) and surface encoder (.pkl) or uses manually specified paths.
- **Interactive Prompt**: Asks the user to paste a JSON object containing the features for a hypothetical match.
- **Prediction**: Uses the loaded model to calculate the win probability for Player 1.
- **Output**: Prints the predicted winner and their win probability to the console.

### 3. Event Scraper (`event_scraper.py`)

**Purpose:**
Scrapes upcoming ATP singles matches for the current and next day from Sofascore.

**How to Run:**
```bash
# First, ensure playwright browsers are installed
playwright install

# Then run the script
python event_scraper.py
```

**Functionality:**
- **Scraping**: Uses playwright to launch a headless browser and fetch data from the internal Sofascore API, bypassing common scraping blocks.
- **Filtering**: Parses events and filters for ATP Men's Singles only, excluding Challenger, ITF, UTR, WTA, juniors, and doubles matches.
- **Output**: Saves the filtered list of upcoming matches to `sofascore_filtered_matches.csv`.

### 4. ELO Fetcher (`elo_fetcher.py`)

**Purpose:**
Fetches the complete ATP ELO rating directory from tennisabstract.com, which serves as a master list of players and their profile page URLs.

**How to Run:**
```bash
python elo_fetcher.py
```

**Functionality:**
- **Data Fetching**: Downloads the HTML page containing the main ELO ratings table.
- **Parsing**: Uses BeautifulSoup and pandas to parse the HTML table and extract player data.
- **URL Extraction**: Correctly constructs absolute profile URLs from the relative links on the page.
- **Output**: Saves the ELO directory to `atp_elo_directory.csv`.

### 5. Player Profile Data Scraper (`generate_player_tables.py`)

**Purpose:**
For each unique player found in `sofascore_filtered_matches.csv`, this script visits their Tennis Abstract profile page and scrapes detailed statistical tables and biographical information. This structured data is ideal for feature engineering or detailed analysis.

**How to Run:**
```bash
# First, ensure playwright browsers are installed
playwright install

# Then run the script
python generate_player_tables.py
```

**Functionality:**
- **Data Loading**: Reads `sofascore_filtered_matches.csv` and `atp_elo_directory.csv`.
- **Player Mapping**: Finds upcoming players and maps them to their profile URLs.
- **Data Extraction**: Uses playwright to:
  - Scrape numerous statistical tables by their HTML ID (e.g., recent-results, career-splits, head-to-heads).
  - Extract player bio data (Age, Handedness, Rank) directly from JavaScript variables on the page.

**Output:**
- Creates a main output directory (`player_profiles_data/`).
- Inside, it creates a subdirectory for each player (e.g., `Carlos_Alcaraz/`).
- Each scraped table is saved as a separate `.csv` file (e.g., `recent-results_1.csv`).
- The player's bio is saved as `player_bio.json`.

### 6. Player Profile PDF Generator (`generate_player_pdfs.py`)

**Purpose:**
An alternative data gathering script that fetches the Tennis Abstract profile pages for players and saves them as PDF files. This is useful for archiving or offline visual analysis of player stats.

**How to Run:**
```bash
# First, ensure playwright browsers are installed
playwright install

# Then run the script
python generate_player_pdfs.py
```

**Functionality:**
- **Data Loading**: Reads `sofascore_filtered_matches.csv` and `atp_elo_directory.csv`.
- **Player Mapping**: Maps upcoming players to their profile URLs.
- **PDF Generation**: Uses crawl4ai (a playwright wrapper) to navigate to each player's profile URL and save the fully rendered page as a PDF.
- **Output**: Saves PDF files (e.g., `Carlos_Alcaraz.pdf`) into the `player_profiles_pdf/` directory.
