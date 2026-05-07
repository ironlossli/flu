#!/usr/bin/env python
import os
import sys
import yaml
import argparse
import shutil
from pathlib import Path
import pandas as pd

# Add src and scripts to python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))

# Import predictor
from predictor.core import SpectraPredictor

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Automated Screening Workflow")
    parser.add_argument("--config", type=str, default="configs/screening.yaml", help="Path to the screening config file")
    args = parser.parse_args()

    # 1. Load Configuration
    config_path = args.config
    print(f"📖 Loading configuration from {config_path}...")
    cfg = load_config(config_path)

    # Prepare Output Directories
    root_out = cfg['output']['root_dir']
    os.makedirs(root_out, exist_ok=True)
    
    # Sub-directories for steps
    pred_output_dir = os.path.join(root_out, "predictions")
    os.makedirs(pred_output_dir, exist_ok=True)

    # ==========================================================================
    # Step 1: Prepare Input
    # ==========================================================================
    print("\n" + "="*50)
    print("📋 Step 1: Preparing Input Data")
    print("="*50)
    
    sel_cfg = cfg['selection']
    input_csv = sel_cfg['input_csv']
    
    if not os.path.exists(input_csv):
        print(f"❌ Input file {input_csv} not found.")
        sys.exit(1)

    # ==========================================================================
    # Step 2: Spectral Prediction
    # ==========================================================================
    print("\n" + "="*50)
    print("🔮 Step 2: Running Spectral Prediction")
    print("="*50)
    
    pred_cfg = cfg['prediction']
    
    predictor = SpectraPredictor.from_checkpoint(
        checkpoint_path=pred_cfg['model_ckpt'],
        device=pred_cfg.get('device', 'cuda'),
        target=pred_cfg.get('target', 'abs')
    )
    
    output_csv = os.path.join(pred_output_dir, "screening_results.csv")
    
    # Update input data with solvent info if needed
    df = pd.read_csv(input_csv)
    if 'solvent_smiles' not in df.columns:
        df['solvent_name'] = pred_cfg['solvent_name']
        df['solvent_smiles'] = pred_cfg['solvent_smiles']
    
    # Limit samples if requested
    if sel_cfg.get('num_samples'):
        df = df.head(sel_cfg['num_samples'])
    
    temp_input = os.path.join(root_out, "temp_input.csv")
    df.to_csv(temp_input, index=False)

    results = predictor.predict_from_file(
        input_path=temp_input,
        output_path=output_csv,
        batch_size=pred_cfg.get('batch_size', 8)
    )

    # ==========================================================================
    # Summary & Cleanup
    # ==========================================================================
    print("\n" + "="*50)
    print("✅ Workflow Completed Successfully!")
    print("="*50)
    
    # Simple Analysis
    if results is not None and not results.empty:
        success_count = (results['status'] == 'success').sum()
        print(f"Processed: {len(results)} molecules")
        print(f"Success:   {success_count}")
        print(f"Results saved to: {output_csv}")
        
        # Display top 5
        val_col = 'wavelength' if 'wavelength' in results.columns else 'prediction'
        if val_col in results.columns:
            print("\nTop 5 Results:")
            print(results.head(5))

    # Cleanup temp file
    if os.path.exists(temp_input):
        os.remove(temp_input)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
