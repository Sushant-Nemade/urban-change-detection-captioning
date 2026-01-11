############  Adds copying of label_rgb images for each new swapped pair.
# Verifies the existence of label_rgb files when checking JSON validity.
# Retains the fallback to CSV if openpyxl is missing.
# Skips augmentation if the existing JSON and all files (A, B, label, label_rgb) are valid.


import json
import os
import sys
import shutil
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
from preprocess_data import tokenize

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
if len(sys.argv) < 2:
    default_env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(default_env_path):
        logging.info(f"No .env path provided; using default: {default_env_path}")
        load_dotenv(default_env_path)
    else:
        raise ValueError("No .env file path provided, and default .env file not found. Please run the script with: python augment_dataset.py path/to/.env")
else:
    load_dotenv(sys.argv[1])
    logging.info(f"Using .env file: {sys.argv[1]}")

# Get environment variables
api_key = os.environ.get('AZURE_OPENAI_API_KEY')
openai_rsn = os.environ.get('AZURE_OPENAI_API_RESOURCE')
llm_deployment = os.environ.get('LLM_DEPLOYMENT', 'gpt-5-2025-08-07')

if not api_key or not openai_rsn:
    raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_API_RESOURCE not set in .env file")

api_base = f"https://{openai_rsn}.openai.azure.com/"
api_version = "2024-08-01-preview"

logging.info(f"Deployment: {llm_deployment}")

# Set up AzureOpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=api_base
)

# Define paths
DATA_PATH_ROOT = r"C:\Users\nemade\Downloads\LEVIR-MCI-data"
input_json = f"{DATA_PATH_ROOT}\\Levir-MCI-dataset\\LevirCCcaptions.json"
output_json = f"{DATA_PATH_ROOT}\\Levir-MCI-dataset\\LevirCCcaptions_augmented.json"
image_dir = f"{DATA_PATH_ROOT}\\Levir-MCI-dataset\\images"

# Clean up redundant files if they exist
def clean_redundant_files(max_filename_nums):
    """Remove augmented files from previous runs if they exist."""
    splits = ["train", "val", "test"]
    for split in splits:
        max_num = max_filename_nums.get(split, 0)
        for folder in ["A", "B", "label", "label_rgb"]:
            folder_path = os.path.join(image_dir, split, folder)
            if not os.path.exists(folder_path):
                continue
            for file in os.listdir(folder_path):
                if file.startswith(f"{split}_") and file.endswith(".png"):
                    try:
                        file_num = int(file.split("_")[1].split(".")[0])
                        if file_num > max_num:
                            file_path = os.path.join(folder_path, file)
                            os.remove(file_path)
                            logging.info(f"Removed redundant file: {file_path}")
                    except ValueError:
                        continue
    if os.path.exists(output_json):
        os.remove(output_json)
        logging.info(f"Removed old JSON: {output_json}")
    if os.path.exists("swapped_pairs.xlsx"):
        os.remove("swapped_pairs.xlsx")
        logging.info("Removed old Excel: swapped_pairs.xlsx")
    if os.path.exists("swapped_pairs.csv"):
        os.remove("swapped_pairs.csv")
        logging.info("Removed old CSV: swapped_pairs.csv")

# Load the original JSON
with open(input_json, "r") as f:
    data = json.load(f)

# Find max filename numbers for each split
max_filename_nums = {}
for split in ["train", "val", "test"]:
    split_entries = [e for e in data["images"] if e["split"] == split]
    if split_entries:
        max_num = max(int(e["filename"].split("_")[1].split(".")[0]) for e in split_entries)
        max_filename_nums[split] = max_num
    else:
        max_filename_nums[split] = 0

# Check if augmented JSON exists and is valid
augmented_valid = False
existing_filenames = set()
if os.path.exists(output_json):
    with open(output_json, "r") as f:
        augmented_data = json.load(f)
    # Verify all new files exist
    new_entries = [e for e in augmented_data["images"] if e["imgid"] >= 10077]
    all_files_exist = True
    for entry in new_entries:
        split = entry["split"]
        filename = entry["filename"]
        for folder in ["A", "B", "label", "label_rgb"]:
            folder_path = os.path.join(image_dir, split, folder)
            if not os.path.exists(folder_path):
                logging.warning(f"Folder {folder_path} does not exist. Marking JSON as invalid.")
                all_files_exist = False
                break
            if not os.path.exists(os.path.join(folder_path, filename)):
                all_files_exist = False
                break
        existing_filenames.add((split, filename))
    if all_files_exist:
        augmented_valid = True
        logging.info("Existing augmented JSON and files (including label_rgb) are valid. Generating only swapped_pairs.xlsx.")
    else:
        logging.warning("Existing augmented JSON found, but some files are missing. Overwriting recommended.")

# Prompt user if JSON exists
if os.path.exists(output_json) and not augmented_valid:
    response = input(f"Augmented JSON {output_json} exists but may be incomplete. Overwrite and clean redundant files? (y/n): ")
    if response.lower() == 'y':
        clean_redundant_files(max_filename_nums)
    else:
        print("Exiting to avoid overwriting existing files. Please manually clean up or use existing JSON.")
        sys.exit(1)

# Initialize counters
imgid_counter = 10077
sentid_counter = 50385

new_entries = []
swapped_pairs = []

# GPT-4o prompt for caption reversal
prompt_template = """Rewrite the following caption for swapped pre- and post-images in remote sensing change detection. Additions (e.g., buildings or roads appearing) should become removals, and removals should become additions. Vary phrasing to avoid repetition.

Examples:
Original: "a building was added."
Swapped: "a building was removed."
Original: "some trees are removed and houses are built."
Swapped: "houses are removed and some trees are added."
Original: "a lane appears on the left and several houses are built at the bottom."
Swapped: "a lane is removed from the left and several houses are removed from the bottom."

Original caption: "{}"
Swapped caption:"""

def simple_reverse_caption(original_caption):
    """Fallback function to reverse caption if API call fails."""
    if "added" in original_caption:
        return original_caption.replace("added", "removed")
    elif "appears" in original_caption:
        return original_caption.replace("appears", "is removed")
    elif "built" in original_caption:
        return original_caption.replace("built", "removed")
    elif "removed" in original_caption:
        return original_caption.replace("removed", "added")
    elif "disappear" in original_caption:
        return original_caption.replace("disappear", "appear")
    else:
        return original_caption

# Skip augmentation if JSON is valid
if not augmented_valid:
    for entry in data["images"]:
        if entry["changeflag"] == 1:
            split = entry["split"]
            original_filename = entry["filename"]
            max_num = max_filename_nums[split]
            new_num = max_num + 1
            max_filename_nums[split] = new_num
            new_filename = f"{split}_{new_num:06d}.png"

            # Check if new filename already exists
            if (split, new_filename) in existing_filenames:
                logging.info(f"File {new_filename} already exists in {split}. Skipping.")
                continue

            # Copy images
            src_pre = os.path.join(image_dir, split, "A", original_filename)
            src_post = os.path.join(image_dir, split, "B", original_filename)
            src_label = os.path.join(image_dir, split, "label", original_filename)
            src_label_rgb = os.path.join(image_dir, split, "label_rgb", original_filename)
            dst_pre = os.path.join(image_dir, split, "A", new_filename)
            dst_post = os.path.join(image_dir, split, "B", new_filename)
            dst_label = os.path.join(image_dir, split, "label", new_filename)
            dst_label_rgb = os.path.join(image_dir, split, "label_rgb", new_filename)

            try:
                shutil.copy(src_post, dst_pre)  # Post becomes pre
                shutil.copy(src_pre, dst_post)  # Pre becomes post
                shutil.copy(src_label, dst_label)  # Copy label
                if os.path.exists(src_label_rgb):
                    shutil.copy(src_label_rgb, dst_label_rgb)  # Copy label_rgb
                else:
                    logging.warning(f"label_rgb file {src_label_rgb} not found. Skipping copy.")
            except Exception as e:
                logging.error(f"Failed to copy images for {original_filename}: {e}")
                continue

            # Reverse captions
            new_sentences = []
            for sentence in entry["sentences"]:
                original_caption = sentence["raw"]
                try:
                    completion = client.chat.completions.create(
                        model=llm_deployment,
                        messages=[
                            {"role": "user", "content": prompt_template.format(original_caption)}
                        ],
                        max_tokens=100,
                        temperature=0.7
                    )
                    if completion.choices and completion.choices[0].message.content:
                        reversed_caption = completion.choices[0].message.content.strip()
                    else:
                        logging.warning(f"Empty response for caption: {original_caption}. Using fallback.")
                        reversed_caption = simple_reverse_caption(original_caption)
                except Exception as e:
                    logging.error(f"API call failed for caption: {original_caption}. Error: {e}")
                    reversed_caption = simple_reverse_caption(original_caption)

                tokens = tokenize(reversed_caption, add_start_token=False, add_end_token=False)
                new_sentences.append({
                    "tokens": tokens,
                    "raw": reversed_caption,
                    "imgid": imgid_counter,
                    "sentid": sentid_counter
                })
                sentid_counter += 1

            # Create new entry
            new_entry = {
                "filepath": split,
                "filename": new_filename,
                "imgid": imgid_counter,
                "sentences": new_sentences,
                "split": split,
                "changeflag": 1,
                "sentids": [s["sentid"] for s in new_sentences]
            }
            new_entries.append(new_entry)
            imgid_counter += 1

            # Track swapped pairs
            swapped_pairs.append({
                "Split": split,
                "Original Filename": original_filename,
                "New Filename": new_filename,
                "Original imgid": entry["imgid"],
                "New imgid": new_entry["imgid"]
            })

    # Append new entries to the original data
    data["images"].extend(new_entries)

    # Save the updated JSON
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)
else:
    # Load swapped pairs from existing JSON
    with open(output_json, "r") as f:
        augmented_data = json.load(f)
    for entry in augmented_data["images"]:
        if entry["imgid"] >= 10077:
            original_entry = next((e for e in data["images"] if e["filename"] in entry["filename"]), None)
            if original_entry:
                swapped_pairs.append({
                    "Split": entry["split"],
                    "Original Filename": original_entry["filename"],
                    "New Filename": entry["filename"],
                    "Original imgid": original_entry["imgid"],
                    "New imgid": entry["imgid"]
                })

# Save swapped pairs to Excel
try:
    df = pd.DataFrame(swapped_pairs)
    df.to_excel("swapped_pairs.xlsx", index=False)
    print("Swapped pairs saved to 'swapped_pairs.xlsx'.")
except ModuleNotFoundError:
    logging.error("Module 'openpyxl' not found. Please install it using 'pip install openpyxl'.")
    print("Saving swapped pairs as CSV instead.")
    df.to_csv("swapped_pairs.csv", index=False)
    print("Swapped pairs saved to 'swapped_pairs.csv'.")
except Exception as e:
    logging.error(f"Failed to save Excel file: {e}")
    print("Saving swapped pairs as CSV instead.")
    df.to_csv("swapped_pairs.csv", index=False)
    print("Swapped pairs saved to 'swapped_pairs.csv'.")

print("Dataset augmentation complete.")