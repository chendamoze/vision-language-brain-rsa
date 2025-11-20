# Visual vs. Abstract Representations in High-Level Visual Cortex

The project investigates whether the organization of the high-level visual cortex (Ventral vs. Dorsal streams) aligns more closely with **visual-descriptive** language or **abstract-conceptual** language. Using Representational Similarity Analysis (RSA), we compared human neural data (fMRI and MEG) against three computational models:
1.  **DCNN-Image:** Visual embeddings (CLIP).
2.  **LLM-Visual Text:** Generated detailed visual descriptions (SGPT embeddings).
3.  **LLM-Abstract Text:** Conceptual descriptions (SGPT embeddings).

## Project Structure

### 1. Text Generation & Optimization
* **`txt_generator.py`**: Generates detailed visual descriptions for images using an iterative feedback loop with **GPT-4o**. It optimizes descriptions to maximize visual similarity (CLIP score) while maintaining semantic distinctiveness from abstract definitions (SGPT score).
* **`best_clip.py`**: Analyzes the iterations from the generator, filters out self-referential errors, and selects the description with the highest CLIP score for the final analysis.

### 2. Main Analysis (RSA)
* **`brain_corr.py`**: The central driver script.
    * Computes Representational Dissimilarity Matrices (RDMs) for all three models (Image, Visual Text, Abstract Text).
    * Calculates correlations between the models.
    * Orchestrates the comparison with neural data (fMRI and MEG).
    * Generates UMAP plots and correlation matrices.

### 3. Neural Data plotting
* **`fmri.py`**: Handles fMRI data analysis. Computes zero-order and partial correlations for Early, Ventral, and Dorsal visual streams, performs Wilcoxon signed-rank tests (FDR-corrected), and plots bar charts.
* **`meg.py`**: Handles MEG data analysis. Performs time-resolved correlation analysis (millisecond resolution), identifies significant time windows, and plots time-course data.

---

## AI Models used
* **GPT-4o (OpenAI):** Used for generating detailed visual descriptions via an iterative feedback loop.
* **CLIP (OpenAI):** Specifically `clip-vit-base-patch32`. Used for:
    * Extracting image embeddings (DCNN-Image).
    * Computing visual similarity scores during the text optimization process.
* **SGPT (Sentence-BERT):** Specifically `Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit`. Used for extracting semantic embeddings from both visual and abstract textual descriptions.




## Installation & Requirements

This project requires Python 3.8+. Install the dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels pingouin torch transformers sentence-transformers openai scikit-learn
```
### Core Libraries
PyTorch, Hugging Face Transformers, Sentence-Transformers, OpenAI API

### Data Analysis & Statistics
* **Pandas & NumPy:** Data manipulation, matrix operations, and RDM construction.
* **SciPy:** Used for Pearson correlations and Wilcoxon signed-rank tests.
* **Pingouin:** Used for computing partial correlations (controlling for covariates).
* **Statsmodels:** Used for False Discovery Rate (FDR) correction (Benjamini-Hochberg).
* **Scikit-learn:** Used for calculating cosine similarity matrices.

### Visualization
* **Matplotlib & Seaborn:** Used for generating bar charts, correlation matrices, and time-course plots.

## Data Availability

To reproduce the results, you need the dataset from Cichy et al. (2016). Source: Cichy, R. M., Pantazis, D., & Oliva, A. (2016). Similarity-based fusion of MEG and fMRI reveals spatio-temporal dynamics in human cortex during visual object recognition.

# Usage

Ensure your project folder contains the following:
imgs_png/ - Directory containing the 92 stimulus images (1.png to 92.png)
semantic_descriptions.json - JSON file with abstract descriptions for the images
all_roi_rdms_small_res.csv - Pre-computed fMRI RDMs (Upper triangle vectorized)
upper_triangle_meg.csv - Pre-computed MEG RDMs (Upper triangle vectorized, time-resolved)

Run the scripts in the following order:

1. Generate Visual Descriptions
(Optional if best_descriptions.json already exists)

```Bash
py txt_generator.py
```

2. Select Best Descriptions
```Bash
py best_clip.py
```

3. Run RSA and Neural Analysis
```Bash
py brain_corr.py
```

