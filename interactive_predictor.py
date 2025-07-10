### ======================================================= ###
###   INTERACTIVE TENNIS MATCH PREDICTION CELL FOR COLAB    ###
###        (FIXED JSON Serialization Error)               ###
### ======================================================= ###

import pandas as pd
import xgboost as xgb
import pickle
import json
import os

# --- 1. Configuration: Updated with the filenames from your image ---
MANUAL_MODEL_PATH = "tennis_model_gpu_complete_20250707_204740.json"
MANUAL_ENCODER_PATH = "surface_encoder_complete_20250707_204740.pkl"


# --- 2. Helper Functions (with the fix) ---

def find_latest_files():
    """Finds the most recently created model and encoder files in the current directory."""
    models = sorted([f for f in os.listdir() if f.startswith('tennis_model_gpu_complete') and f.endswith('.json')])
    encoders = sorted([f for f in os.listdir() if f.startswith('surface_encoder_complete') and f.endswith('.pkl')])

    if not models or not encoders:
        return None, None

    return models[-1], encoders[-1]

def load_model_and_encoder(model_path, encoder_path):
    """Loads the trained XGBoost model and the surface encoder from disk."""
    print(f"🔄 Loading model from: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("✅ Model loaded successfully.")

    print(f"🔄 Loading encoder from: {encoder_path}")
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print("✅ Encoder loaded successfully.")

    return model, encoder

def predict_match(model, input_data: dict):
    """Predicts the outcome of a tennis match from the structured JSON input."""
    features = input_data["features"]
    context = input_data["match_context"]
    p1_name = context["player_1"]

    input_df = pd.DataFrame([features])
    model_feature_names = model.get_booster().feature_names
    input_df = input_df[model_feature_names]

    win_prob_p1 = model.predict_proba(input_df)[0][1]

    if win_prob_p1 >= 0.5:
        favorite_player = context["player_1"]
        predicted_probability = win_prob_p1
    else:
        favorite_player = context["player_2"]
        predicted_probability = 1 - win_prob_p1

    # ================================================================= #
    # ▼▼▼ THE FIX IS HERE ▼▼▼
    # We explicitly convert the numpy.float32 to a standard Python float
    # before rounding and adding it to the dictionary.
    # ================================================================= #
    result = {
        "predicted_win_probability": round(float(predicted_probability), 4),
        "favorite_player": favorite_player
    }
    return result

# --- 3. Main Interactive Execution ---

def main():
    # This is the example data structure. You can copy this.
    EXAMPLE_JSON_STRING = '''
    {
      "match_context": {
        "player_1": "Carlos Alcaraz",
        "player_2": "Jannik Sinner",
        "surface": "Grass",
        "tournament_level": "Unknown",
        "match_id": "alcaraz_vs_sinner_grass_simulation"
      },
      "features": {
        "glicko_confidence_diff": 0.15, "glicko_win_prob": 0.55, "glicko_diff": 38,
        "rank_diff": 1, "rank_pts_diff": -1130, "rd_diff": 0.08, "surf_elo_diff": 173,
        "ranking_volatility": -0.12, "form_confidence": 0.18, "freshness_penalty": -0.05,
        "age_diff": 1, "is_grand_slam": 0, "h2h_adv": 0.56, "rd_uncertainty_flag": 0,
        "min_rd": 45.2, "max_rd": 52.8, "surface_experience_diff": 8, "recent_form_diff": 0.22,
        "confidence_weighted_elo": 2267, "height_diff": 0, "draw_size_log": 4.09,
        "activity_diff": -0.08, "confidence_product": 0.82, "elo_momentum": 15.2,
        "elo_diff": 38, "rust_factor": 0.02, "form_diff": 0.18, "is_masters": 0,
        "surface_encoded": 2, "hand_adv": 0, "career_1st_win_diff": 0.024,
        "career_ace_diff": 0.011, "career_df_diff": -0.008, "career_1st_serve_diff": 0.003
      }
    }
    '''

    try:
        # --- SETUP: Load the model and encoder ---
        print("--- MODEL SETUP ---")
        if MANUAL_MODEL_PATH and MANUAL_ENCODER_PATH:
            print("Using manually specified file paths.")
            model_path, encoder_path = MANUAL_MODEL_PATH, MANUAL_ENCODER_PATH
        else:
            print("Attempting to find latest files automatically...")
            model_path, encoder_path = find_latest_files()

        if not model_path or not encoder_path:
            raise FileNotFoundError("Could not find model/encoder files.")

        tennis_model, surface_encoder = load_model_and_encoder(model_path, encoder_path)
        print("-" * 20)

        # --- INTERACTIVE PREDICTION LOOP ---
        print("\n🎾 TENNIS PREDICTOR IS READY 🎾")
        print("Instructions:")
        print("1. Prepare your match data in the JSON format below.")
        print("2. Paste the entire JSON object into the input box and press Enter.")
        print("3. To finish, type 'exit' and press Enter.")
        print("\n--- JSON TEMPLATE (you can copy this) ---")
        print(EXAMPLE_JSON_STRING)
        print("-------------------------------------------\n")

        while True:
            # Prompt user for multi-line input
            print("\n👇 Paste your match data JSON here and press Enter (or type 'exit' to quit):")
            user_input_str = input()

            if user_input_str.strip().lower() == 'exit':
                print("👋 Exiting predictor. Goodbye!")
                break

            if not user_input_str.strip():
                print("⚠️ Input is empty. Please paste the JSON data or type 'exit'.")
                continue

            try:
                # Parse the user's string input into a Python dictionary
                input_data = json.loads(user_input_str)

                # Get the prediction
                prediction_result = predict_match(tennis_model, input_data)

                # Display the result
                print("\n" + "="*50)
                print("MATCH CONTEXT")
                print(f"  Player 1: {input_data['match_context']['player_1']}")
                print(f"  Player 2: {input_data['match_context']['player_2']}")
                print(f"  Surface: {input_data['match_context']['surface']}")
                print("="*50)
                print("\n🏆 MODEL PREDICTION OUTPUT 🏆")
                print(json.dumps(prediction_result, indent=2))
                print("="*50)

            except json.JSONDecodeError:
                print("\n❌ ERROR: Invalid JSON format. Please check your pasted text.")
                print("Common mistakes include missing commas, brackets {}, or quotes \"\".")
            except KeyError as e:
                print(f"\n❌ ERROR: The provided JSON is missing a required key: {e}")
                print("Please ensure 'match_context' and 'features' keys are present.")
            except Exception as e:
                print(f"\n❌ An unexpected error occurred: {e}")


    except FileNotFoundError:
        print("\n" + "!"*50)
        print("🔥 ERROR: MODEL OR ENCODER FILE NOT FOUND! 🔥")
        print(f"The script was looking for '{MANUAL_MODEL_PATH}' and '{MANUAL_ENCODER_PATH}'.")
        print("Please make sure these files exist in your Colab session directory.")
        print("You may need to run the training cell again.")
        print("!"*50)

if __name__ == "__main__":
    main()
