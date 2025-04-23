import torch  # >=1.6.0
import os
import pandas as pd
import numpy as np
import tqdm
import glob
import sys
from PIL import Image
import torchvision.transforms as transforms
import argparse

from normalization import CenterCropNoPad, get_list_norm
from normalization2 import PaddingWarp
from get_method_here import get_method_here, def_model

from sklearn.metrics import f1_score, precision_score, accuracy_score

def evaluate_results(output_dir, model_names):
    all_results = {}
    
    # First, collect all predictions and labels
    for model_name in model_names:
        all_results[model_name] = {
            'all_labels': [],
            'all_preds': [],
            'video_labels': [],
            'video_preds': []
        }
    
    # Gather predictions from all datasets
    for dataset_dir in os.listdir(output_dir):
        dataset_path = os.path.join(output_dir, dataset_dir)
        csv_path = os.path.join(dataset_path, f"{dataset_dir}.csv")
        if not os.path.isfile(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        
        if 'label' not in df.columns:
            continue
            
        labels = df['label'].astype(int).values
        
        # Collect data for each model
        for model_name in model_names:
            if model_name not in df.columns:
                continue
                
            # Get framewise predictions
            preds = (df[model_name] > 0.5).astype(int).values
            all_results[model_name]['all_labels'].extend(labels)
            all_results[model_name]['all_preds'].extend(preds)
            
            # Process videowise predictions
            df['video'] = df['src'].apply(lambda x: x.split('/')[1])  # Adjust if structure differs
            
            for vid, group in df.groupby('video'):
                pred_vals = (group[model_name] > 0.5).astype(int).values
                label_vals = group['label'].astype(int).values
                
                majority_pred = int(np.round(pred_vals.mean()))
                majority_label = int(np.round(label_vals.mean()))
                
                all_results[model_name]['video_preds'].append(majority_pred)
                all_results[model_name]['video_labels'].append(majority_label)
    
    # Calculate unified metrics for each model
    for model_name in model_names:
        if not all_results[model_name]['all_labels']:
            print(f"No data found for model: {model_name}")
            continue
            
        # Convert lists to numpy arrays
        all_labels = np.array(all_results[model_name]['all_labels'])
        all_preds = np.array(all_results[model_name]['all_preds'])
        video_labels = np.array(all_results[model_name]['video_labels'])
        video_preds = np.array(all_results[model_name]['video_preds'])
        
        print(f"\n🔍 UNIFIED METRICS | Model: {model_name}")
        
        # --- FRAMEWISE METRICS ---
        acc_f = accuracy_score(all_labels, all_preds)
        prec_f = precision_score(all_labels, all_preds)
        f1_f = f1_score(all_labels, all_preds)
        
        print("🧩 Framewise:")
        print(f"   Accuracy:  {acc_f:.4f}")
        print(f"   Precision: {prec_f:.4f}")
        print(f"   F1-score:  {f1_f:.4f}")
        
        # --- VIDEOWISE METRICS ---
        acc_v = accuracy_score(video_labels, video_preds)
        prec_v = precision_score(video_labels, video_preds)
        f1_v = f1_score(video_labels, video_preds)
        
        print("🎬 Videowise:")
        print(f"   Accuracy:  {acc_v:.4f}")
        print(f"   Precision: {prec_v:.4f}")
        print(f"   F1-score:  {f1_v:.4f}")
        
        # You could also add per-dataset metrics if needed
        print("\n📊 Per-dataset metrics:")
        for dataset_dir in os.listdir(output_dir):
            dataset_path = os.path.join(output_dir, dataset_dir)
            csv_path = os.path.join(dataset_path, f"{dataset_dir}.csv")
            if not os.path.isfile(csv_path):
                continue
                
            df = pd.read_csv(csv_path)
            
            if 'label' not in df.columns or model_name not in df.columns:
                continue
                
            labels = df['label'].astype(int).values
            preds = (df[model_name] > 0.5).astype(int).values
            
            print(f"  Dataset: {dataset_dir}")
            print(f"    Accuracy:  {accuracy_score(labels, preds):.4f}")

def running_tests(data_path, output_dir, weights_dir, csv_file, batch_size=16):
    DATA_PATH = data_path

    print("CURRENT OUT FOLDER")
    print(output_dir)
    datasets = {os.path.basename(os.path.dirname(_)): _ for _ in glob.glob(DATA_PATH + "*/")}
    csvfilename = csv_file
    outroot = output_dir

    if not os.path.exists(outroot):
        os.makedirs(outroot)

    print(f"Found {len(datasets)} datasets:", datasets.keys())

    # Auto-select device
    device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"----> Using device: {device}")

    # List of models
    models_list = {
        'Corvi_pretrain': 'Corvi_pretrain'
    }

    models_dict = dict()
    transform_dict = dict()
    for model_name in models_list:
        _, model_path, arch, norm_type, patch_size = get_method_here(models_list[model_name], weights_path=weights_dir)

        model = def_model(arch, model_path, localize=False)
        model = model.to(device).eval()

        transform = list()

        # Define a fixed size for all images (e.g., 224x224)
        fixed_size = (224, 224)

        if patch_size is not None:
            if isinstance(patch_size, tuple):
                print('input resize:', patch_size)
                transform.append(transforms.Resize(*patch_size))
                transform_key = f'res{patch_size[0]}_{norm_type}'
            else:
                if patch_size > 0:
                    print('input crop:', patch_size)
                    transform.append(CenterCropNoPad(patch_size))
                    transform_key = f'crop{patch_size}_{norm_type}'
                else:
                    print('input crop pad:', patch_size)
                    transform.append(CenterCropNoPad(-patch_size))
                    transform.append(PaddingWarp(-patch_size))
                    transform_key = f'cropp{-patch_size}_{norm_type}'
        else:
            transform_key = f'none_{norm_type}'

        # Add resizing to the fixed size
        transform.append(transforms.Resize(fixed_size))

        transform = transform + get_list_norm(norm_type)
        transform = transforms.Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)

    print("Available transforms:", list(transform_dict.keys()))
    print("Available models:", list(models_dict.keys()))

    # Read main CSV file
    main_table = pd.read_csv(csvfilename)[['src']]

    # Process each dataset with batch processing
    with torch.no_grad():
        for dataset in datasets:
            outdir = os.path.join(outroot, dataset)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            print(f"\n=== Processing dataset: {dataset} ===")
            output_csv = os.path.join(outdir, f"{dataset}.csv")
            rootdataset = DATA_PATH
            
            # Filter table for this dataset
            dataset_table = main_table[main_table['src'].str.contains(dataset + "/")].copy()
            print(f"{dataset}: Found {len(dataset_table)} entries in CSV")
            
            # Validate frames and collect paths
            valid_frames = []
            for index, dat in dataset_table.iterrows():
                if dataset in dat['src'].split('/')[0]:
                    full_path = os.path.join(rootdataset, dat['src'])
                    if os.path.isfile(full_path):
                        valid_frames.append((index, full_path))
            
            # Create DataFrame with only valid frames
            valid_indices = [idx for idx, _ in valid_frames]
            table_to_save = dataset_table.loc[valid_indices].copy()
            print(f"Found {len(valid_frames)} valid frames out of {len(dataset_table)} entries")
            
            # Check if we need to process anything
            do_models = list(models_dict.keys())
            if os.path.isfile(output_csv):
                existing_table = pd.read_csv(output_csv)
                do_models = [m for m in models_dict.keys() if m not in existing_table.columns]
                
                if 'src' in existing_table.columns and len(existing_table) == len(table_to_save):
                    # Check if we can reuse the existing table
                    if all(a == b for a, b in zip(existing_table['src'], table_to_save['src'])):
                        table_to_save = existing_table
                    else:
                        # Columns don't match, create new
                        do_models = list(models_dict.keys())
                else:
                    # Different lengths, create new
                    do_models = list(models_dict.keys())
                    
            do_transforms = set([models_dict[m][0] for m in do_models])
            print(f"Models to process: {do_models}")
            print(f"Transforms to use: {do_transforms}")

            if len(do_models) == 0 or len(valid_frames) == 0:
                print(f"Skipping dataset {dataset} - no work to do")
                continue

            # Process in batches
            num_frames = len(valid_frames)
            for batch_start in tqdm.tqdm(range(0, num_frames, batch_size), desc=f"Processing {dataset}"):
                batch_end = min(batch_start + batch_size, num_frames)
                batch_frames = valid_frames[batch_start:batch_end]
                batch_id = [idx for idx, _ in batch_frames]
                
                # Prepare image batches for each transform type
                batch_images = {k: [] for k in do_transforms}
                
                # Load and transform images
                for _, filename in batch_frames:
                    img = Image.open(filename).convert('RGB')
                    for k in do_transforms:
                        batch_images[k].append(transform_dict[k](img))
                
                # Process each model with appropriate transform
                for model_name in do_models:
                    transform_key = models_dict[model_name][0]
                    model = models_dict[model_name][1]
                    
                    # Stack images into a batch tensor
                    img_batch = torch.stack(batch_images[transform_key], 0).to(device)
                    
                    # Run model on batch
                    out_tens = model(img_batch).cpu().numpy()
                    
                    # Process output
                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False, f"Unexpected output shape: {out_tens.shape}"
                        
                    if len(out_tens.shape) > 1:
                        logits = np.mean(out_tens, (1, 2))
                    else:
                        logits = out_tens
                    
                    # Store results
                    for i, idx in enumerate(batch_id):
                        table_to_save.loc[idx, model_name] = logits[i]
            
            # Add label column
            if "real" in dataset:
                table_to_save.insert(1, 'label', False)
            else:
                table_to_save.insert(1, 'label', True)
                
            # Save results
            table_to_save.to_csv(output_csv, index=False)
            print(f"Saved results to {output_csv}")


def main():
    print("Running the Tests")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, 
                        help="Path to the images of the testset", 
                        default=os.path.join(os.path.dirname(__file__), "dataset/test/test_set_1/"))
    parser.add_argument("--out_dir", type=str, 
                        help="Path where the output CSVs should be saved", 
                        default=os.path.join(os.path.dirname(__file__), "results_test"))
    parser.add_argument("--weights_dir", type=str, 
                        help="Path to the network weights", 
                        default=os.path.join(os.path.dirname(__file__), "weights"))
    parser.add_argument("--csv_file", type=str, 
                        help="Path to the CSV file", 
                        default=os.path.join(os.path.dirname(__file__), "operations.csv"))
    parser.add_argument("--batch_size", type=int, 
                        help="Batch size for processing", 
                        default=8)
    
    args = vars(parser.parse_args())
    running_tests(
        args['data_dir'], 
        args['out_dir'], 
        args['weights_dir'], 
        args['csv_file'],
        args['batch_size']
    )

    # Evaluate results
    evaluate_results(args['out_dir'], ['Corvi_pretrain'])


if __name__ == "__main__":
    main()