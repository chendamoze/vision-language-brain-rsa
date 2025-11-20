import os
import json
import base64
import numpy as np
import pandas as pd
from PIL import Image
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
import torch
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


ITERATIONS = 11
NUM_IMGS = 93

def convert_types(obj):
    """
    Converts NumPy scalar types (e.g., np.int32, np.float64) to native Python types
    so they can be JSON serialized. Raises a TypeError for unsupported types.
    """
    if isinstance(obj, np.generic):
        return obj.item()  
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    
def get_image_embedding(image_path, model, processor):
    """
    Generates an image embedding using the CLIP model.
    """
    image = Image.open(image_path)
    # Preprocess the image and convert it to PyTorch tensors
    inputs = processor(images=image, return_tensors="pt")
    # Disable gradient calculation since we're only doing inference
    with torch.no_grad():
        # Extract image features (embedding) using the CLIP model
        return model.get_image_features(**inputs)


def get_text_embedding(text, model, processor):
    """
    Generates a text embedding using the CLIP model.
    """
    # Preprocess the input text and convert it to PyTorch tensors, with padding if needed
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        return model.get_text_features(**inputs)


def compute_clip_similarity(image_path, text, model, processor):
    """
    Computes cosine similarity between image and text embeddings using CLIP.
    """
    image_emb = get_image_embedding(image_path, model, processor)
    text_emb = get_text_embedding(text, model, processor)
    return cosine_similarity(image_emb.cpu().numpy(), text_emb.cpu().numpy())[0][0]


def compute_sgpt_similarity(D_v, D_s, sgpt_model):
    """
    Computes cosine similarity between a visual description and semantic description using SGPT.
    """
    # Encode both descriptions into embeddings using the SGPT model
    embeddings = sgpt_model.encode([D_v, D_s])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def build_feedback(best_visual, prev_visual, best_semantic, prev_semantic, D_v, best_D_v):
    """
    Constructs detailed feedback for the next visual description generation based on similarity scores.
    """
    feedback_parts = []

    # The four possible cases based on a comparison between current and previous
    if best_visual > prev_visual and best_semantic < prev_semantic:
        feedback_parts.append("Excellent progress! Your new description captures the image more clearly and also moves further away from simply repeating the semantic version. If you'd like to refine it further, you might explore more vivid or precise details in the scene.")

    elif best_visual > prev_visual and best_semantic >= prev_semantic:
        feedback_parts.append("You're improving! The description reflects the image better than before. However, it still sounds similar to the original semantic phrasing. Try rewording your sentence using fresh expressions that stay grounded in the image.")

    elif best_visual <= prev_visual and best_semantic < prev_semantic:
        feedback_parts.append("Nice work making the description more semantically distinct—it feels fresher and more original. Now, try to focus on enhancing the connection to the visual content itself. Think about what stands out in the image and build around that.")

    elif D_v == best_D_v or (best_visual == prev_visual and best_semantic == prev_semantic):
        feedback_parts.append("There doesn't seem to be a clear improvement in either visual or semantic aspects. Consider re-approaching the description from a different angle—maybe highlight a new object, emotion, or dynamic in the image that wasn't mentioned before.")

    elif best_visual <= prev_visual and best_semantic >= prev_semantic:
        feedback_parts.append("The current description appears to be less visually aligned and more semantically similar than before. This may be a good opportunity to take a fresh look at the image and describe it using your own perspective—what's unique about it? What catches your eye?")

    return " ".join(feedback_parts)

def gpt_step_with_feedback(image_path, client, previous_description=None, feedback=None):
    """
    Calls OpenAI API to generate a visual-only description of an image based on previous description and feedback.
    """
    # Prepare base messages
    messages = [{"role": "system", "content": "Please create a detailed visual description focusing only on what can be observed. Describe the visual elements as they appear without naming or identifying specific objects or subjects. No longer than 76 characters."}]


    # Add previous description and feedback as separate text messages if available
    if previous_description:
        messages.append({"role": "user", "content": f"Previous description: {previous_description}"})

    if feedback:
        messages.append({"role": "user", "content": f"Feedback: {feedback}"})

    # Read and encode image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Add the image as a separate message (using base64 encoding for PNG)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"
                }
            }
        ]
    })

   
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=messages
    )

    # Return the new description
    return response.choices[0].message.content

def compute_similarity_matrices(dicts, sgpt_model):
    """
    Computes cosine similarity matrices for visual and semantic descriptions.
    
    Returns:
        ids (list): Ordered list of IDs.
        visual_sim_matrix (ndarray): Similarity matrix for visual descriptions.
        semantic_sim_matrix (ndarray): Similarity matrix for semantic descriptions.
    """
    # Ensure consistent ordering
    ids = sorted(dicts.keys())
    descriptions = [dicts[id] for id in ids]
    
    # Calculate embeddings
    embeddings = sgpt_model.encode(descriptions)
    
    # Compute similarity matrices
    sim_matrix = cosine_similarity(embeddings)

    return ids, sim_matrix

def compute_correlation(visual_sim_matrix, semantic_sim_matrix, output_dir, iteration, ids):
    """
    Creates and saves similarity matrices for visual descriptions, semantic descriptions, 
    and calculates correlation between them.
    """
    
    # Calculate correlation between the two matrices
    # Flatten the upper triangular part of both matrices (excluding diagonal)
    n = len(ids)
    visual_flat = []
    semantic_flat = []
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangular
            visual_flat.append(visual_sim_matrix[i, j])
            semantic_flat.append(semantic_sim_matrix[i, j])
    
    # Calculate Pearson and Spearman correlations
    pearson_corr, pearson_p = pearsonr(visual_flat, semantic_flat)
    spearman_corr, spearman_p = spearmanr(visual_flat, semantic_flat)
    
    # Save correlation results
    correlation_data = {
        "iteration": iteration,
        "pearson_correlation": float(pearson_corr),
        "pearson_p_value": float(pearson_p),
        "spearman_correlation": float(spearman_corr),
        "spearman_p_value": float(spearman_p)
    }
    output_dir_corr = os.path.join("./results/correlations")
    os.makedirs(output_dir_corr, exist_ok=True)
    with open(os.path.join(output_dir_corr, f"matrix_correlation_{iteration}.json"), 'w') as f:
        json.dump(correlation_data, f, indent=2, default=convert_types)
    
def plot_scores_per_image(plotting_data, output_dir):
    """
    Plots CLIP score vs. SGPT score for each image across all iterations.
    Saves individual plots for each image.
    plotting_data: Dict where key is image_id (str) and value is a list of (iteration, clip_score, sgpt_score) tuples.
    """
    plot_output_dir = os.path.join(output_dir, "score_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    for img_id, iter_data_list in plotting_data.items():
        if not iter_data_list: # Skip if no data for this image
            continue

        # Extract iterations, clip_scores, and sgpt_scores from the list of tuples
        iterations = [item[0] for item in iter_data_list]
        clip_scores = [item[1] for item in iter_data_list]
        sgpt_scores = [item[2] for item in iter_data_list]

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, clip_scores, marker='o', linestyle='-', color='blue', label='CLIP Score (Visual)')
        plt.plot(iterations, sgpt_scores, marker='x', linestyle='--', color='red', label='SGPT Score (Semantic)')

        plt.title(f'CLIP vs. SGPT Scores for Image {img_id} Across Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_output_dir, f'image_{img_id}_scores.png'))
        plt.close()
    
def plot_average_scores_per_iteration(plotting_data, output_dir):

    plot_output_dir = os.path.join(output_dir, "avg_score_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    sgpt_avg = {i:sum(plotting_data[i]["sgpt_score"])/(NUM_IMGS-1) for i in range(1, ITERATIONS)}
    clip_avg = {i:sum(plotting_data[i]["clip_score"])/(NUM_IMGS-1)  for i in range(1, ITERATIONS)}
    iterations = list(sgpt_avg.keys())
    sgpt_values = list(sgpt_avg.values())
    clip_values = list(clip_avg.values())

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, clip_values, label="CLIP (Visual)", marker='o', color='blue')
    plt.plot(iterations, sgpt_values, label="SGPT (Semantic Distance)", marker='x', color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Average Score")
    plt.title("Average CLIP and SGPT Scores per Iteration")
    plt.xticks(iterations)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, "average_scores_per_iteration.png"))
    plt.close()
     

def run_iteration(clip_model, clip_processor, sgpt_model, output_dir, semantic_descriptions, semantic_sim_matrix, client):
    plotting_data = {str(i): [] for i in range(1, NUM_IMGS)} # Initialize with empty lists for each image
    plotting_avg = {j: {"clip_score" : [], "sgpt_score":[]} for j in range(1, ITERATIONS)}
    prev_desc = {} # "i":{"D_v":__, "clip_score":__, "sgpt_score":__}
    best_score = {} # "i":{"D_v":__, "clip_score":__, "sgpt_score":__}
    for iteration in range(1, ITERATIONS):
        curr_desc = {}
        for img_i in range(1, NUM_IMGS):
            i = str(img_i)
            image_path = f"./imgs_png/{img_i}.png"
            # Generate new visual description based on the previous and the best 
            if iteration > 1:
                best_visual = best_score[i]["clip_score"]
                prev_visual = prev_desc[i]["clip_score"]
                best_semantic = best_score[i]["sgpt_score"]
                prev_semantic = prev_desc[i]["sgpt_score"]
                D_v = prev_desc[i]["D_v"]
                best_D_v = best_score[i]["D_v"]
                feedback = build_feedback(best_visual, prev_visual, best_semantic, prev_semantic, D_v, best_D_v)
                new_Dv = gpt_step_with_feedback(image_path, client, D_v, feedback)
            
            else:
                new_Dv = gpt_step_with_feedback(image_path, client)

            # Update the best score according to clip ascore
            D_s = semantic_descriptions[i]
            clip_score = compute_clip_similarity(image_path, new_Dv, clip_model, clip_processor)
            sgpt_score = compute_sgpt_similarity(new_Dv, D_s, sgpt_model)
            if i not in best_score or best_score[i]["clip_score"] < clip_score:
                best_score[i] = {"D_v" : new_Dv, "clip_score" : clip_score, "sgpt_score" : sgpt_score}
            
            curr_desc[i] = {"D_v" : new_Dv, "clip_score" : clip_score, "sgpt_score" : sgpt_score}
            plotting_data[i].append((iteration, clip_score, sgpt_score))
            plotting_avg[iteration]["clip_score"].append(clip_score)
            plotting_avg[iteration]["sgpt_score"].append(sgpt_score)
            
        
        # Save all the visual descriptions for the current iteration
        output_dir_itr = os.path.join("./results/iterations")
        os.makedirs(output_dir_itr, exist_ok=True)
        with open(os.path.join(output_dir_itr, f"iteration_{iteration}.json"), 'w') as f:
            json.dump(curr_desc, f, indent=2, default=convert_types)
        
        ids, visual_sim_matrix = compute_similarity_matrices(curr_desc, sgpt_model)
        visual_df = pd.DataFrame(visual_sim_matrix, index=ids, columns=ids)
        output_dir_rdm = os.path.join("./results/RDMs")
        os.makedirs(output_dir_rdm, exist_ok=True)
        visual_df.to_csv(os.path.join(output_dir_rdm, f"RDM_visual_{iteration}.csv"))
        compute_correlation(visual_sim_matrix, semantic_sim_matrix, output_dir, iteration, ids)
        
        prev_desc = curr_desc
        print(f"iteration {iteration}")
    # Call plotting function with the pre-organized data
    plot_scores_per_image(plotting_data, output_dir)
    plot_average_scores_per_iteration(plotting_avg, output_dir)


        
            


def main():

    api_key = "your-api-key"
    client = OpenAI(api_key=api_key)
    device= torch.device('cpu')
    sgpt_model = SentenceTransformer("Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit",device="cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Create the result folder
    output_dir = os.path.join("./results")
    os.makedirs(output_dir, exist_ok=True)

    semantic_file = os.path.join("./semantic_descriptions.json")
    with open(semantic_file, 'r') as f:
        semantic_descriptions = json.load(f)
    
    ids, semantic_sim_matrix = compute_similarity_matrices(semantic_descriptions, sgpt_model)
    semantic_df = pd.DataFrame(semantic_sim_matrix, index=ids, columns=ids)
    semantic_df.to_csv(os.path.join(output_dir, f"semantic_similarity_matrix.csv"))

    run_iteration(clip_model, clip_processor, sgpt_model, output_dir, semantic_descriptions, semantic_sim_matrix, client)


if __name__ == "__main__":
    main()
