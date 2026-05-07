
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import load_predictor, load_data

def extract_embeddings(predictor, df):
    """
    Run inference on dataframe and collect embeddings.
    """
    embeddings = []
    predictions = []
    valid_indices = []

    print("Extracting embeddings...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            res = predictor.predict(
                xyz_path=row['xyz_path'],
                solvent_smiles=row['solvent_smiles'],
                return_features=True
            )
            # Use 'graph_embedding' as the representation
            emb = res['graph_embedding'] 
            # If graph_embedding is empty or invalid, skip
            if emb.size == 0: continue
            
            # Flatten if necessary (though usually it's [H])
            embeddings.append(emb.flatten())
            predictions.append(res['wavelength'])
            valid_indices.append(idx)
        except Exception as e:
            # print(f"Skipping index {idx}: {e}")
            pass
            
    return np.array(embeddings), np.array(predictions), df.iloc[valid_indices]

def plot_tsne(embeddings, values, output_path):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    X_embedded = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=values, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(sc, label='Predicted Wavelength (nm)')
    plt.title("t-SNE of Solute Graph Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved t-SNE plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Latent Space (t-SNE)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV (xyz_path, solvent_smiles)")
    parser.add_argument("--output", type=str, default="tsne_plot.png", help="Output image path")
    parser.add_argument("--limit", type=int, default=500, help="Max samples to process")
    args = parser.parse_args()

    predictor = load_predictor(args.ckpt)
    df = load_data(args.input_csv)
    
    if args.limit and len(df) > args.limit:
        print(f"Limiting to first {args.limit} samples.")
        df = df.iloc[:args.limit]

    embeddings, predictions, _ = extract_embeddings(predictor, df)
    
    if len(embeddings) < 5:
        print("Not enough successful embeddings to run t-SNE.")
        return

    plot_tsne(embeddings, predictions, args.output)

if __name__ == "__main__":
    main()
