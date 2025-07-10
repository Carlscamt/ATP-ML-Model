# ATP Match Prediction Model Suite

This repository contains a suite of Python scripts for training a tennis match prediction model, making predictions, and gathering relevant data.

## Scripts

### 1. Model Training (`model_training.py`)

**Purpose:**
This is the core script for training the XGBoost-based tennis match prediction model. It handles everything from data acquisition to model evaluation and serialization.

**How to Run:**
```bash
python model_training.py
```

**Functionality:**
- **Data Acquisition:** Downloads historical ATP match data (2010-2024) from Jeff Sackmann's public repository.
- **Feature Engineering:** Processes the raw data to create a rich feature set, including:
  - ELO and Glicko-2 ratings.
  - Surface-specific ELO ratings.
  - Head-to-head matchup history.
  - Player form, activity, and freshness metrics.
  - Physical attributes (age, height, handedness).
  - Career statistics.
- **Training:** 
  - Automatically detects and utilizes a GPU for significantly faster training if `nvidia-smi` is available, otherwise falls back to CPU.
  - Splits the data into training and validation sets.
  - Trains an XGBoost classifier.
- **Evaluation:** Reports the model's accuracy and AUC score on the validation set.
- **Output:** Saves the following files:
  - `tennis_model_gpu_complete_{timestamp}.json`: The trained XGBoost model.
  - `surface_encoder_complete_{timestamp}.pkl`: The scikit-learn `LabelEncoder` for court surfaces.
  - `feature_importance_complete_{timestamp}.csv`: A report of the most influential features.
  - `tennis_feature_frame.pkl`: The full, processed data frame used for feature creation.

---

### 2. Interactive Predictor (`interactive_predictor.py`)

**Purpose:**
Provides a command-line interface to make predictions on a single, user-defined match using a pre-trained model.

**How to Run:**
```bash
python interactive_predictor.py
```

**Functionality:**
- **Model Loading:** Automatically loads the most recently trained model (`.json`) and surface encoder (`.pkl`) from the directory.
- **Interactive Prompt:** Asks the user to paste a JSON object containing the features for a hypothetical match.
- **Prediction:** Uses the loaded model to calculate the win probability for Player 1.
- **Output:** Prints the predicted winner and their win probability to the console.

---

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
- **Scraping:** Uses `playwright` to launch a headless browser and fetch data from the internal Sofascore API.
- **Filtering:** Parses the events and filters out non-ATP matches, such as Challenger, ITF, UTR, WTA, juniors, and doubles events.
- **Output:** Saves the filtered list of upcoming matches to `sofascore_filtered_matches.csv`.

---

### 4. ELO Fetcher (`elo_fetcher.py`)

**Purpose:**
Fetches the complete ATP ELO rating directory from `tennisabstract.com`.

**How to Run:**
```bash
python elo_fetcher.py
```

**Functionality:**
- **Data Fetching:** Downloads the HTML page containing the main ELO ratings table.
- **Parsing:** Uses `BeautifulSoup` and `pandas` to parse the HTML table and extract player data.
- **URL Extraction:** Appends the direct `tennisabstract.com` profile URL for each player.
- **Output:** Saves the entire ELO directory to `atp_elo_directory.csv`.
