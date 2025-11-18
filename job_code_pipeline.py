# Make sure you ran cjc.py, and all datasets are updated in the AI Catalog

# Refresh Datasets:
    # Prediction_CMB_6M_V2_515
    # Prediction_CMB_12M_V2_515
    
    # Prediction_TRA_6M_V2_515
    # Prediction_TRA_12M_V2_515
    
    # Prediction_ALL_VARIANTS_6M_V2_515
    # Prediction_ALL_VARIANTS_12M_V2_515

# Upload to Projects & Generate Predictions:
    # Prediction_CMB_6M_V2_515 -> DR Project Primary_CMB_500
    # Prediction_CMB_12M_V2_515 -> DR Project Primary_CMB_500
    
    # Prediction_TRA_6M_V2_515 -> DR Project Primary_TRA_500
    # Prediction_TRA_12M_V2_515 -> DR Project Primary_TRA_500
    
    # Prediction_ALL_VARIANTS_6M_V2_515 -> DR Project Primary_ALL_500
    # Prediction_ALL_VARIANTS_12M_V2_515 -> DR Project Primary_ALL_500

# Download Predictions from DR

# Run this script using this command: python3 job_code_pipeline.py

import os
import shutil
import glob
import pandas as pd
import time
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# === Step 1: Move & Rename Downloaded Files ===
downloads_dir = os.path.expanduser("~/Downloads")
target_dir = os.getenv("JOB_CODE_TARGET_DIR")
timeframes = ["6M", "12M"]
model_types = ["CMB", "TRA", "ALL"]

for timeframe in timeframes:
    for model in model_types:
        model_suffix = f"{model}_VARIANTS" if model == "ALL" else model
        pattern = os.path.join(
            downloads_dir,
            f"Primary_{model}_*_Prediction_{model_suffix}_{timeframe}_V2_515.csv"
        )
        matches = glob.glob(pattern)

        if not matches:
            print(f"❌ No matching file found for {model} {timeframe}")
            continue

        latest_file = max(matches, key=os.path.getctime)
        new_filename = f"Primary_{model}_500_{timeframe}_V2_515.csv"
        destination_path = os.path.join(target_dir, new_filename)
        shutil.move(latest_file, destination_path)
        print(f"✅ Moved and renamed: {latest_file} → {destination_path}")

# === Step 2: Combine Predictions ===
def process_file(input_file, range_value):
    print(f"\U0001F4E5 Processing: {input_file}")
    start = time.time()
    df = pd.read_csv(input_file)

    df_long = df.melt(
        id_vars=["VIN_NO", "MODEL", "Variant", "Creation_Date"],
        value_vars=[col for col in df.columns if col.startswith("Prediction ")],
        var_name="JOB_CODE",
        value_name="PROBABILITY"
    )

    df_long["JOB_CODE"] = df_long["JOB_CODE"].str.replace("Prediction ", "", regex=False)
    df_long = df_long[df_long["PROBABILITY"] >= 0.1]

    df_top5 = (
        df_long
        .sort_values(["VIN_NO", "PROBABILITY"], ascending=[True, False])
        .groupby("VIN_NO")
        .head(5)
        .reset_index(drop=True)
    )
    df_top5["Predictive_Range"] = range_value
    print(f"✅ Extracted top 5: {len(df_top5):,} rows ⏱️ {time.time() - start:.2f}s\n")
    return df_top5

def run_pipeline(suffix):
    print(f"\U0001F501 Running prediction pipeline for: {suffix.upper()}")
    range_value = 6 if suffix == "6M" else 12
    cmb = process_file(f"Primary_CMB_500_{suffix}_V2_515.csv", range_value)
    tra = process_file(f"Primary_TRA_500_{suffix}_V2_515.csv", range_value)
    all_df = process_file(f"Primary_ALL_500_{suffix}_V2_515.csv", range_value)
    combined = pd.concat([cmb, tra, all_df], ignore_index=True)
    print(f"📦 Combined {suffix.upper()} total: {len(combined):,} rows\n")
    return combined

combined_6m = run_pipeline("6M")
combined_12m = run_pipeline("12M")

final_combined = pd.concat([combined_6m, combined_12m], ignore_index=True)
final_combined['Creation_Date'] = pd.to_datetime(
    final_combined['Creation_Date'].astype(str).str.strip(), errors='coerce'
)
final_combined["JOB_CODE_upper"] = final_combined["JOB_CODE"].str.upper()
final_combined = final_combined.drop_duplicates(subset=["VIN_NO", "JOB_CODE_upper", "Creation_Date"])
final_combined = final_combined.drop(columns=["JOB_CODE_upper"])

final_combined = final_combined[[
    "VIN_NO", "MODEL", "Variant", "Creation_Date", "Predictive_Range", "JOB_CODE", "PROBABILITY"
]]

output_file = "JOB_CODE_Combined_Filtered.csv"
final_combined.to_csv(output_file, index=False)
print(f"✅ Final combined data saved to: {output_file} ({len(final_combined):,} rows)")

# === Step 3: Move Files to a Dated Folder ===
today = date.today()
year = str(today.year)
month = f"{today.month:02d}"
day = f"{today.day:02d}"

archive_dir = os.path.join(target_dir, year, month, day)
os.makedirs(archive_dir, exist_ok=True)

for filename in os.listdir(target_dir):
    file_path = os.path.join(target_dir, filename)
    if (
        os.path.isdir(file_path)
        or filename.endswith(".py")
        or filename.endswith(".yaml")
        or filename.startswith(".")
        or file_path == archive_dir
    ):
        continue
    shutil.move(file_path, os.path.join(archive_dir, filename))
    print(f"✅ Moved: {filename} → {archive_dir}/")

# All files should be moved to the todays dated folder, excluding .py scripts

# Upload JOB_CODE_Combined_Filtered.csv to Azure at raw-datarobot-push/JobCode

# All done!
