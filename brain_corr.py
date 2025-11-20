
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import partial_corr, corr
from meg import meg_plot
from fmri import fmri_plot, create_upper_triangle_comb
from best_clip import save_best_visual
import ast



def convert_types(obj):
    """
    Converts NumPy scalar types to native Python types so they can be serialized to JSON.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compute_rdm_from_descriptions(filepath, model, excluded=None, k=None, output_csv=None):
    """
    Computes an RDM from text descriptions
    using a sentence-transformer model, and saves the upper triangle as CSV.
    """

    with open(filepath, 'r') as f:
        data = json.load(f)
    ids = sorted([i for i in data.keys() if i not in excluded], key=lambda x: int(x))
    # Extract the relevant text descriptions
    if not k:
        descriptions = [data[id] for id in ids]
    else:
        descriptions = [data[id][k] for id in ids]

    # Generate embeddings
    embeddings = model.encode(descriptions, batch_size=8)
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    # Convert to RDM
    rdm = 1 - sim_matrix
    # Extract the upper triangular values (excluding the diagonal)
    triu_indices = np.triu_indices(rdm.shape[0], k=1)
    dnn_vec = rdm[triu_indices]

    # Save to CSV if output path provided
    if output_csv:
        df = pd.DataFrame({
            'image1': [ids[j] for j in triu_indices[1]],
            'image2': [ids[i] for i in triu_indices[0]],
            'score': dnn_vec
        })
        df.to_csv(output_csv, index=False)

    return dnn_vec


def compute_image_rdm(image_paths, model, processor, device='cpu'):
    """
    Computes an image-based RDM using CLIP image embeddings.
    """

    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
    embeddings = embeddings.cpu().numpy()
    # Compute RDM
    sim_matrix = cosine_similarity(embeddings)
    rdm = 1 - sim_matrix
    output_csv = "./saved_dnns/image_rdm.csv"
    # Get upper triangle values
    triu_indices = np.triu_indices(rdm.shape[0], k=1)
    dnn_vec = rdm[triu_indices]

    # Generate image IDs from file names
    ids = [path+1 for path in range(len(image_paths))]

    # Save to CSV
    df = pd.DataFrame({
        'image1': [ids[j] for j in triu_indices[1]],  
        'image2': [ids[i] for i in triu_indices[0]],  
        'score': dnn_vec
    })
    df.to_csv(output_csv, index=False)
    
    return dnn_vec

def csv_to_dnn(filepath):
    """
    Load a DNN model vector from CSV where each row contains a 'score' column.
    """
    df = pd.read_csv(filepath)
    vector = df['score'].to_numpy()
    return vector

def compute_and_save_correlations(dnn_image_vec, dnn_visual_text_vec, dnn_abstract_text_vec, save_dir="correlation_results"):
    """
    Computes zero-order and partial correlations between three DNN vectors:
    image DNN, visual-text DNN, and abstract-text DNN. Saves results to CSV.
    """
    os.makedirs(save_dir, exist_ok=True)

    data = pd.DataFrame({
        'Image': dnn_image_vec,
        'Visual Text': dnn_visual_text_vec,
        'Abstract Text': dnn_abstract_text_vec
    })

    results = []
    # Define comparison pairs and the covariate (control variable) for partial correlation
    pairs = [
        ('Image', 'Visual Text', 'Abstract Text'),
        ('Image', 'Abstract Text', 'Visual Text'),
        ('Visual Text', 'Abstract Text', 'Image')
    ]

    for x, y, covar in pairs:
        # Zero-order correlation (Pearson)
        zero_order_res = corr(data[x], data[y], method='pearson').iloc[0]
        # Partial correlation (Pearson)
        partial_res = partial_corr(data=data, x=x, y=y, covar=covar, method='pearson').iloc[0]

        results.append({
            'Var1': x,
            'Var2': y,
            'Control (for partial)': covar,
            'Zero-order_r': zero_order_res['r'],
            'Zero-order_p': zero_order_res['p-val'],
            'Partial_r': partial_res['r'],
            'Partial_p': partial_res['p-val']
        })

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "dnn_correlations.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved DNN correlations to {csv_path}")
    return csv_path

def p_to_stars(p_val):
    """
    Convert a p-value into common significance star notation.
    """

    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return 'n.s.'

def add_bracket(ax, x1, x2, y, height, text):
    """
    Draws a significance bracket on a Matplotlib axis.
    """
    line_x = [x1, x1, x2, x2]
    line_y = [y, y + height, y + height, y]
    ax.plot(line_x, line_y, lw=1.5, color='black')
    ax.text((x1 + x2) * 0.5, y + height, text, ha='center', va='bottom', color='black', fontsize=14)

def plot_correlations_styled(csv_path, corr_type='Partial', save_path=None):
    """
    Creates a styled correlation bar plot (Partial or Zero-order), including
    significance stars and condition labels, and saves the figure.
    """
    if corr_type not in ['Partial', 'Zero-order']:
        raise ValueError("corr_type must be 'Partial' or 'Zero-order'")

    df = pd.read_csv(csv_path)
    
    r_col = f'{corr_type}_r'
    p_col = f'{corr_type}_p'
    
    plot_order = [
        ('Image', 'Visual Text'),
        ('Image', 'Abstract Text'),
        ('Visual Text', 'Abstract Text')
    ]
    
    plot_data = []
    for var1, var2 in plot_order:
        row = df[((df['Var1'] == var1) & (df['Var2'] == var2)) | ((df['Var1'] == var2) & (df['Var2'] == var1))]
        if not row.empty:
            plot_data.append(row.iloc[0])
    
    plot_df = pd.DataFrame(plot_data)
    values = plot_df[r_col].values
    p_values = plot_df[p_col].values

    fig, ax = plt.subplots(figsize=(8, 7))
    bar_positions = np.arange(len(plot_df))
    bars = ax.bar(bar_positions, values, width=0.7, color='#cccccc', edgecolor='black')

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        significance = p_to_stars(p_values[i])
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, significance, ha='center', va='bottom', fontsize=16, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval / 2, f'{yval:.2f}', ha='center', va='center', fontsize=16, color='black')

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.spines['bottom'].set_visible(False)

    y_pos_line1, y_pos_line2, y_pos_line3 = -0.08, -0.13, -0.18
    # column 1
    ax.text(0, y_pos_line1, 'Image', color='blue', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(0, y_pos_line2, '&', color='black', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(0, y_pos_line3, 'Visual Text', color='purple', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    # column 2
    ax.text(1, y_pos_line1, 'Image', color='blue', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(1, y_pos_line2, '&', color='black', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(1, y_pos_line3, 'Abstract Text', color='red', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    # column 3
    ax.text(2, y_pos_line1, 'Visual Text', color='purple', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(2, y_pos_line2, '&', color='black', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(2, y_pos_line3, 'Abstract Text', color='red', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    

    max_val = max(values)
    bracket_y_base = max_val + 0.1
    bracket_height = 0.03
    add_bracket(ax, 0, 1, bracket_y_base + 0.1, bracket_height, '***')
    add_bracket(ax, 1, 2, bracket_y_base, bracket_height, '***')
    add_bracket(ax, 0, 2, bracket_y_base + 0.2, bracket_height, '***')

 
    ax.set_ylabel(f'{corr_type} correlation between DNNs', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max_val + 0.35)
    plt.subplots_adjust(bottom=0.2) 

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()




   
def main():

    
    excluded = save_best_visual()

    # Load Models
    device = torch.device('cpu')
    print("Models...")
    sgpt_model = SentenceTransformer("Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit", device="cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Compute RDMs
    print("Creating dnns...")
    save_dir = "saved_dnns"
    os.makedirs(save_dir, exist_ok=True)
    # Visual text RDM
    dnn_visual_vec = compute_rdm_from_descriptions("./best_descriptions.json", sgpt_model, excluded=excluded, k="D_v", output_csv="./saved_dnns/best_visual_rdm.csv")
    
    # Semantic text RDM
    dnn_semantic_vec = compute_rdm_from_descriptions("./semantic_descriptions.json", sgpt_model, excluded=excluded, output_csv="./saved_dnns/semantic_rdm.csv")
    
    # Image RDM
    image_paths = [f"./imgs_png/{i}.png" for i in range(1, 93) if str(i) not in excluded]
    dnn_image_vec = compute_image_rdm(image_paths, clip_model, processor)

    # Compute and plot DNN to DNN Correlations
    csv_file_path = compute_and_save_correlations(
        dnn_image_vec=dnn_image_vec,
        dnn_visual_text_vec=dnn_visual_vec,
        dnn_abstract_text_vec=dnn_semantic_vec
    )

    plot_correlations_styled(
        csv_path=csv_file_path,
        corr_type='Partial',
        save_path='correlation_results/partial_correlations_plot.png'
    )

    plot_correlations_styled(
        csv_path=csv_file_path,
        corr_type='Zero-order',
        save_path='correlation_results/zero_order_correlations_plot.png'
    )
    
    # fMRI Analysis (Brain-Model Correlations)   
    print("Create fmri upper...")
    # Create RDM vectors (upper triangle) for each subject-ROI pair from the fMRI data
    fmri_matrices = create_upper_triangle_comb("./all_roi_rdms_small_res.csv", ["subj", "roi"], "fMRI", excluded)
    print("=== fMRI plotting ===")
    fmri_plot(fmri_matrices, dnn_image_vec, dnn_semantic_vec, dnn_visual_vec)

    # MEG Analysis (Brain-Model Correlations over Time)
    print("Create meg upper...")
    # Create RDM vectors (upper triangle) for each subj-session-ms from the MEG data
    df = pd.read_csv("./upper_triangle_meg.csv")
    meg_matrices = {}

    for _, row in df.iterrows():
        # Convert string "(1, 1, -100)" to tuple
        key = ast.literal_eval(row['group'])
        # Collect all values in value_1 ... value_3741
        values = [row[col] for col in df.columns if col.startswith('value')]
        meg_matrices[key] = values

    print("=== MEG plotting ===")
    meg_plot(meg_matrices, dnn_image_vec, dnn_semantic_vec, dnn_visual_vec)
    

    

if __name__ == "__main__":
    main()

