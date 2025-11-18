# Run this script using this command: python3 cjc.py

import os
import shutil
import glob
import pandas as pd
import numpy as np
import re
import time
import datarobot as dr
import datetime
import requests
from dotenv import load_dotenv
load_dotenv()

# Initialize DataRobot client
dr_config_path = os.path.join(os.path.dirname(__file__), "drconfig.yaml")
dr.Client(config_path=dr_config_path)

dataset_status_messages = []
errors_detected = []
log_messages = []

# === Check Update Status for Dependent Datasets ===

dependent_dataset_ids = {
    "WkRoFile Dataset": os.getenv("WkRoFile_DATASET_ID"),
    "WKOTHSUB Dataset": os.getenv("WKOTHSUB_DATASET_ID"),
    "WkVehFl Dataset": os.getenv("WkVehFl_DATASET_ID"),
    "Primary Dataset": os.getenv("PRIMARY_DATASET_ID")
}

webhook_url = os.getenv("SLACK_WEBHOOK_URL")

# Store freshness check messages

for name, ds_id in dependent_dataset_ids.items():
    try:
        ds = dr.Dataset.get(ds_id)
        updated_dt = pd.to_datetime(ds.created_at)
        age_days = (pd.Timestamp.now(tz=updated_dt.tz) - updated_dt).days

        status_line = ""
        if age_days > 7:
            status_line = f"❌ {name} last updated {age_days} days ago ({updated_dt.date()})"
            dataset_status_messages.append(status_line)
            errors_detected.append(f"Error checking {name}: Dataset last updated {age_days} days ago")
            print(status_line)
            continue
        else:
            status_line = f"📅 {name} is up to date ({age_days} days ago, {updated_dt.date()})"
            dataset_status_messages.append(status_line)
            print(status_line)
    except Exception as e:
        error_msg = f"⚠️ Failed to check {name}: {e}"
        dataset_status_messages.append(error_msg)
        errors_detected.append(f"Error checking {name}: {e}")
        print(error_msg)
        continue


# === Download Latest Version of Primary Dataset ===

# Initialize client
dr.Client(config_path=dr_config_path)

# Set dataset ID and download destination
source_dataset_id = os.getenv("PRIMARY_DATASET_ID")
input_file_path = 'JOB_CODE_PRIMARY_DATASET_515.csv'

# Get dataset and download the file
try:
    source_dataset = dr.Dataset.get(source_dataset_id)

    print(f"Found dataset: {source_dataset.name}")
    print(f"Dataset ID: {source_dataset.id}")
    print(f"Dataset Version ID: {source_dataset.version_id}")
    print(f"Is latest version: {source_dataset.is_latest_version}")

    with open(input_file_path, 'wb') as f:
        source_dataset.get_file(filelike=f)
    print(f"✅ Downloaded source dataset to: {input_file_path}")
    log_messages.append(f"✅ Downloaded source dataset to: {input_file_path}")
except Exception as e:
    error_msg = f"❌ Failed to download source dataset: {e}"
    errors_detected.append("Failed to download source dataset")
    print(error_msg)

if errors_detected:
    slack_message = {
        "text": "*🚨 Script Stopped Due to:*\n" + "\n".join(dataset_status_messages)
    }
    try:
        slack_response = requests.post(webhook_url, json=slack_message)
        if slack_response.status_code == 200:
            print("📣 Slack notification sent.")
        else:
            print(f"⚠️ Slack notification failed: {slack_response.status_code}")
    except Exception as e:
        print(f"⚠️ Slack notification error: {e}")
    exit(1)


# === Clean Primary Dataset ===

# File paths
input_file_path = 'JOB_CODE_PRIMARY_DATASET_515.csv'
final_output_path = 'JOB_CODE_PRIMARY_DATASET_CLEANED_515.csv'

# Step 1 — Initial cleaning
start = time.time()
df = pd.read_csv(input_file_path)
print(f"✔️ Loaded input file with {len(df):,} rows")

model_prefixes = df['MODEL'].dropna().unique().astype(str)

def clean_job_code_vectorized(job_codes, model_prefixes):
    job_codes = job_codes.str.replace(r'[/_]', '', regex=True)

    for prefix in model_prefixes:
        pattern = r'^' + re.escape(prefix) + r'(?=[A-Z])'
        job_codes = job_codes.str.replace(pattern, '', regex=True)

    remove_keywords_pattern = r'\b(?:TRAVEL|SEGMENT|SEG|DIAG|Gen-|Contact)\b'
    job_codes = job_codes.where(~job_codes.str.contains(remove_keywords_pattern, na=False, case=False, regex=True))

    return job_codes.str.strip().replace('', np.nan)

df['Cleaned_JOB_CODE'] = clean_job_code_vectorized(df['JOB_CODE'], model_prefixes)
df_cleaned = df.dropna(subset=['Cleaned_JOB_CODE'])

print(f"🔢 Rows after Step 1 (initial cleaning): {len(df_cleaned):,}")
print(f"⏱️ Step 1 completed in {time.time() - start:.2f} seconds\n")

# Step 2 — Further filtering
start = time.time()
model_prefixes = df_cleaned['MODEL'].dropna().unique().astype(str)

def clean_job_code_step2(job_codes, model_prefixes):
    job_codes = job_codes.str.replace(r'[/_]', '', regex=True)
    job_codes = job_codes.str.replace(r'^(7-IRON|2WD|AWD|4WD|FWD|RWD|\d{1,4})', '', regex=True)

    for prefix in model_prefixes:
        pattern = r'^' + re.escape(prefix) + r'(?=[A-Z])'
        job_codes = job_codes.str.replace(pattern, '', regex=True)

    remove_keywords_pattern = r'\b(?:TRAVEL|SEGMENT|SEG|DIAG|Gen-|Contact|WARRANTY|SEG)\b|WARRANTYSEG|SEG\d*'
    job_codes = job_codes.where(~job_codes.str.contains(remove_keywords_pattern, na=False, case=False, regex=True))

    return job_codes.str.strip().replace('', np.nan)

df_cleaned.loc[:, 'Cleaned_JOB_CODE'] = clean_job_code_step2(df_cleaned['JOB_CODE'], model_prefixes)

df_cleaned = df_cleaned.copy()
df_cleaned['Cleaned_JOB_CODE_2'] = df_cleaned['Cleaned_JOB_CODE'].where(
    ~df_cleaned['Cleaned_JOB_CODE'].str.contains(r'\bWARRANTY\b|\bSEG\b', na=False, case=False)
)

filtered_df = df_cleaned[~df_cleaned['Cleaned_JOB_CODE_2'].str.contains('DIAG', na=False)]

print(f"🔢 Rows after Step 2 (refined filtering): {len(filtered_df):,}")
print(f"⏱️ Step 2 completed in {time.time() - start:.2f} seconds\n")

# Step 3 — Final filtering and output
start = time.time()
unwanted_pattern = r"(?:seg|promo|wash|gen|warranty|diag|comb)"

df_final = filtered_df.dropna(subset=["Cleaned_JOB_CODE_2"])
df_final = df_final[~df_final["Cleaned_JOB_CODE_2"].str.contains(unwanted_pattern, case=False, na=False)]

df_final = df_final.drop(columns=["JOB_CODE"])
df_final = df_final.rename(columns={"Cleaned_JOB_CODE_2": "JOB_CODE"})
df_final = df_final[["RO_NUMBER", "VIN_NO", "MODEL", "Variant", "REG", "JOB_CODE"]]

df_final.to_csv(final_output_path, index=False)

print(f"🔢 Rows after Step 3 (final output): {len(df_final):,}")
print(f"⏱️ Step 3 completed in {time.time() - start:.2f} seconds\n")
print(f"✅ Clean source dataset saved to: {final_output_path}")
log_messages.append(f"✅ Cleaned source dataset saved to: {final_output_path}")


# === Upload Clean Dataset to DataRobot ===

try:
    dr.Client(config_path=dr_config_path)

    # Dataset details
    CLEANED_DATASET_ID = os.getenv("CLEANED_DATASET_ID")
    FILENAME = final_output_path

    if not os.path.exists(FILENAME):
        raise FileNotFoundError(f"❌ File not found: {FILENAME}")

    print(f"📤 Uploading '{FILENAME}' to DataRobot dataset ID {CLEANED_DATASET_ID}...")

    # Upload the file
    new_version = dr.Dataset.create_version_from_file(
        dataset_id=CLEANED_DATASET_ID,
        file_path=FILENAME,
        max_wait=600
    )

    print(f"✅ Upload complete. New version ID: {new_version.version_id}")
    log_messages.append(f"✅ {FILENAME} was successfully uploaded to DataRobot. New version ID: {new_version.version_id}")
    

except Exception as e:
    print(f"❌ {FILENAME} was not uploaded to DataRobot: {e}")
    log_messages.append(f"❌ {FILENAME} was not uploaded to DataRobot: {e}")


# === Send Slack notification once complete ===

# Prepare dataset status block for Slack message
status_block = "\n".join(dataset_status_messages)
log_block = "\n".join(log_messages)

# Set URLs from .env file
cleaned_dataset_url = os.getenv("CLEANED_DATASET_URL")
vin_dataset_url = os.getenv("VIN_DATASET_URL")
cmb_6m_dataset_url = os.getenv("CMB_6M_DATASET_URL")
cmb_12m_dataset_url = os.getenv("CMB_12M_DATASET_URL")
tra_6m_dataset_url = os.getenv("TRA_6M_DATASET_URL")
tra_12m_dataset_url = os.getenv("TRA_12M_DATASET_URL")
all_6m_dataset_url = os.getenv("ALL_6M_DATASET_URL")
all_12m_dataset_url = os.getenv("ALL_12M_DATASET_URL")

cmb_model_url = os.getenv("CMB_MODEL_URL")
tra_model_url = os.getenv("TRA_MODEL_URL")
all_model_url = os.getenv("ALL_MODEL_URL")

slack_message = {
    "text": (
        #"*✅ Script Completed Successfully*", # LATER: Build logic to check if everything ran correctly
        f"*Dataset Refresh Check:*\n```{status_block}```\n"
        f"{log_block}\n\n\n"
        f"*Dataset Refresh Files:*\n"
        f"🔗 [Cleaned Dataset]\n({cleaned_dataset_url})\n"
        f"🔗 [VIN Dataset]\n({vin_dataset_url})\n"
        f"🔗 [CMB 6M Dataset]\n({cmb_6m_dataset_url})\n"
        f"🔗 [CMB 12M Dataset]\n({cmb_12m_dataset_url})\n"
        f"🔗 [TRA 6M Dataset]\n({tra_6m_dataset_url})\n"
        f"🔗 [TRA 12M Dataset]\n({tra_12m_dataset_url})\n"
        f"🔗 [ALL 6M Dataset]\n({all_6m_dataset_url})\n"
        f"🔗 [ALL 12M Dataset]\n({all_12m_dataset_url})\n\n\n"
        f"*Upload Datasets to Appropriate Models:*\n"
        f"🔗 [CMB Model]\n({cmb_model_url})\n"
        f"🔗 [TRA Model]\n({tra_model_url})\n"
        f"🔗 [ALL Model]\n({all_model_url})\n"
    )
}

try:
    slack_response = requests.post(webhook_url, json=slack_message)
    if slack_response.status_code == 200:
        print("📣 Slack notification sent.")
    else:
        print(f"⚠️ Slack notification failed: {slack_response.status_code}")
except Exception as e:
    print(f"⚠️ Slack notification error: {e}")



# Check that JOB_CODE_PRIMARY_DATASET_CLEANED_515.csv was updated in the AI Catalog
# Check that Primary_Dataset_VIN_NO_515 was updated in the AI Catalog
# Check that CMB 6M & 12M were updated in the AI Catalog
# Check that TRA 6M & 12M were updated in the AI Catalog
# Check that ALL 6M & 12M were updated in the AI Catalog

# Move to the job_code_pipeline.py script