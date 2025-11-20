import os
import json
import pandas as pd

# Constants
ITERATIONS_FOLDER = './results/iterations'
OUTPUT_FILE = 'best_descriptions.json'
# Set of words indicating self-reference to filter out
SELF_REF_WORDS = {"i", "i'm", "me", "my", "mine", "myself"}

def contains_self_reference(text):
    """
    Checks if the text contains any self-referential words.
    Returns True if found, False otherwise.
    """
    if not text:
        return False
    text_lower = text.lower()
    # Using padded spaces to ensure we match whole words only
    return any(f" {word} " in f" {text_lower} " for word in SELF_REF_WORDS)

def save_best_visual():
    """
    Loads all iteration files once, calculates global stats (min/max SGPT),
    filters out self-referential descriptions, and saves the entry with the 
    highest CLIP score for each image.
    """
    all_records = []

    if not os.path.exists(ITERATIONS_FOLDER):
        print(f"Warning: Folder {ITERATIONS_FOLDER} not found.")
        return set()

    # Iterate over files in the directory
    for filename in os.listdir(ITERATIONS_FOLDER):
        if filename.startswith("iteration_") and filename.endswith(".json"):
            filepath = os.path.join(ITERATIONS_FOLDER, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for image_id, info in data.items():
                    # Flatten the structure 
                    record = {
                        'image_id': str(image_id),
                        'D_v': info.get('D_v', ''),
                        'clip_score': info.get('clip_score', 0),
                        'sgpt_score': info.get('sgpt_score', 0)
                    }
                    all_records.append(record)
    
    if not all_records:
        print("No data found.")
        return set()

    # Convert list of dicts to a pandas DataFrame 
    df = pd.DataFrame(all_records)

    # Calculate global statistics 
    stats = df.groupby('image_id')['sgpt_score'].agg(['min', 'max'])

    # Filter invalid descriptions
    # Identify rows containing self-referential words
    df['is_self_ref'] = df['D_v'].apply(contains_self_reference)
    
    # Keep only valid descriptions
    df_clean = df[~df['is_self_ref']].copy()

    # Select Best Descriptions (Highest CLIP Score)
    # Sort by CLIP score descending. 
    best_rows = df_clean.sort_values('clip_score', ascending=False).drop_duplicates('image_id', keep='first')

    output = {}
    for _, row in best_rows.iterrows():
        img_id = row['image_id']
        sgpt_val = row['sgpt_score']
        
        # Check if the chosen score is an extreme value (min or max) relative to all iterations
        img_stats = stats.loc[img_id]
        sgpt_status = None
        if sgpt_val == img_stats['max']:
            sgpt_status = "max"
        elif sgpt_val == img_stats['min']:
            sgpt_status = "min"

        output[img_id] = {
            'D_v': row['D_v'],
            'clip_score': row['clip_score'],
            'sgpt_extreme': sgpt_status
        }

    # Identify excluded images
    # Images that appeared in the data but have no valid descriptions
    all_image_ids = set(df['image_id'].unique())
    included_ids = set(output.keys())
    excluded_images = all_image_ids - included_ids

    # Sort output by image ID before saving
    sorted_output = {k: output[k] for k in sorted(output.keys(), key=lambda x: int(x))}
    
    # Save to JSON file 
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        json.dump(sorted_output, out_f, indent=2, ensure_ascii=False)
    
    print(f"Best descriptions saved to {OUTPUT_FILE}")
    return excluded_images

