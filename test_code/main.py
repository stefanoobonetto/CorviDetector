import torch  # >=1.6.0
import os
import pandas
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
    all_results = []

    for dataset_dir in os.listdir(output_dir):
        dataset_path = os.path.join(output_dir, dataset_dir)
        csv_path = os.path.join(dataset_path, f"{dataset_dir}.csv")
        if not os.path.isfile(csv_path):
            continue

        df = pandas.read_csv(csv_path)

        if 'label' not in df.columns:
            continue

        labels = df['label'].astype(int).values  # convert True/False to 1/0

        for model_name in model_names:
            if model_name not in df.columns:
                continue

            preds = (df[model_name] > 0.5).astype(int).values  # soglia 0.5

            print(f"\n📄 Dataset: {dataset_dir} | Model: {model_name}")

            # --- FRAMEWISE METRICS ---
            acc_f = accuracy_score(labels, preds)
            prec_f = precision_score(labels, preds)
            f1_f = f1_score(labels, preds)

            print("🧩 Framewise:")
            print(f"   Accuracy:  {acc_f:.4f}")
            print(f"   Precision: {prec_f:.4f}")
            print(f"   F1-score:  {f1_f:.4f}")

            # --- VIDEOWISE METRICS ---
            # Assumiamo che il path contenga il nome del video come prima cartella dopo il dataset
            df['video'] = df['src'].apply(lambda x: x.split('/')[1])  # Adatta questo se la struttura è diversa

            video_preds = []
            video_labels = []

            for vid, group in df.groupby('video'):
                pred_vals = (group[model_name] > 0.5).astype(int).values
                label_vals = group['label'].astype(int).values

                majority_pred = int(np.round(pred_vals.mean()))
                majority_label = int(np.round(label_vals.mean()))

                video_preds.append(majority_pred)
                video_labels.append(majority_label)

            acc_v = accuracy_score(video_labels, video_preds)
            prec_v = precision_score(video_labels, video_preds)
            f1_v = f1_score(video_labels, video_preds)

            print("🎬 Videowise:")
            print(f"   Accuracy:  {acc_v:.4f}")
            print(f"   Precision: {prec_v:.4f}")
            print(f"   F1-score:  {f1_v:.4f}")

def runnig_tests(data_path, output_dir, weights_dir, csv_file):
    DATA_PATH = data_path

    print("CURRENT OUT FOLDER")
    print(output_dir)
    datasets = {os.path.basename(os.path.dirname(_)): _ for _ in glob.glob(DATA_PATH + "*/")}
    csvfilename = csv_file
    outroot = output_dir

    if not os.path.exists(outroot):
        os.makedirs(outroot)

    print(len(datasets), datasets.keys())

    # Automatically select device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("----> Using device:", device)

    batch_size = 1

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

        if patch_size is not None:
            if isinstance(patch_size, tuple):
                print('input resize:', patch_size)
                transform.append(transforms.Resize(*patch_size))
                transform_key = 'res%d_%s' % (patch_size[0], norm_type)
            else:
                if patch_size > 0:
                    print('input crop:', patch_size)
                    transform.append(CenterCropNoPad(patch_size))
                    transform_key = 'crop%d_%s' % (patch_size, norm_type)
                else:
                    print('input crop pad:', patch_size)
                    transform.append(CenterCropNoPad(-patch_size))
                    transform.append(PaddingWarp(-patch_size))
                    transform_key = 'cropp%d_%s' % (-patch_size, norm_type)
        else:
            transform_key = 'none_%s' % norm_type

        transform = transform + get_list_norm(norm_type)
        transform = transforms.Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)

    print(list(transform_dict.keys()))
    print(list(models_dict.keys()))

    # Test
    with torch.no_grad():
        table = pandas.read_csv(csvfilename)[['src', ]]
        for dataset in datasets:
            outdir = os.path.join(outroot, dataset)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            print(outdir)
            output_csv = os.path.join(outdir, f"{dataset}.csv")
            rootdataset = DATA_PATH
            
            # Preprocess table for this dataset
            dataset_table = table[table['src'].str.contains(dataset + "/")].copy()
            print(dataset, 'Number of entries in CSV:', len(dataset_table))
            
            # Create a new table with only valid frames
            valid_frames = []
            for index, dat in dataset_table.iterrows():
                if dataset in dat['src'].split('/')[0]:
                    full_path = os.path.join(rootdataset, dat['src'])
                    if os.path.isfile(full_path):
                        valid_frames.append((index, full_path))
            
            # Create a new DataFrame with only valid frames
            valid_indices = [idx for idx, _ in valid_frames]
            table_to_save = dataset_table.loc[valid_indices].copy()
            print(f"Found {len(valid_frames)} valid frames out of {len(dataset_table)} entries")
            
            if os.path.isfile(output_csv):
                existing_table = pandas.read_csv(output_csv)
                do_models = [_ for _ in models_dict.keys() if _ not in existing_table]
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
            else:
                do_models = list(models_dict.keys())
                
            do_transforms = set([models_dict[_][0] for _ in do_models])
            print("Models to process:", do_models)
            print("Transforms to use:", do_transforms)

            if len(do_models) == 0 or len(valid_frames) == 0:
                print(f"Skipping dataset {dataset} - no work to do")
                continue

            batch_img = {k: list() for k in transform_dict}
            batch_id = list()
            
            # Process only valid frames
            for index, filename in tqdm.tqdm(valid_frames, total=len(valid_frames)):
                # Process the frame
                for k in transform_dict:
                    batch_img[k].append(transform_dict[k](Image.open(filename).convert('RGB')))
                batch_id.append(index)

                if len(batch_id) >= batch_size:
                    for k in do_transforms:
                        batch_img[k] = torch.stack(batch_img[k], 0)
                    for model_name in do_models:
                        out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()

                        if out_tens.shape[1] == 1:
                            out_tens = out_tens[:, 0]
                        elif out_tens.shape[1] == 2:
                            out_tens = out_tens[:, 1] - out_tens[:, 0]
                        else:
                            assert False
                        if len(out_tens.shape) > 1:
                            logit1 = np.mean(out_tens, (1, 2))
                        else:
                            logit1 = out_tens

                        for ii, logit in zip(batch_id, logit1):
                            table_to_save.loc[ii, model_name] = logit

                    batch_img = {k: list() for k in transform_dict}
                    batch_id = list()

            if len(batch_id) > 0:
                for k in transform_dict:
                    batch_img[k] = torch.stack(batch_img[k], 0)
                for model_name in models_dict:
                    out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()

                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False
                    if len(out_tens.shape) > 1:
                        logit1 = np.mean(out_tens, (1, 2))
                    else:
                        logit1 = out_tens
                    for ii, logit in zip(batch_id, logit1):
                        table_to_save.loc[ii, model_name] = logit
                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

            if "real" in dataset:
                table_to_save.insert(1, 'label', False)
            else:
                table_to_save.insert(1, 'label', True)
            table_to_save.to_csv(output_csv, index=False)  # Save the results as a CSV file

def main():
    print("Running the Tests")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The path to the images of the testset on which the operations have been pre applied with the provided code", default=os.path.join(os.path.dirname(__file__), "dataset/test/test_set_1/"))
    # parser.add_argument("--data_dir", type=str, help="The path to the images of the testset on which the operations have been pre applied with the provided code", default="")
    parser.add_argument("--out_dir", type=str, help="The Path where the csv containing the outputs of the networks should be saved", default=os.path.join(os.path.dirname(__file__), "results_test"))
    parser.add_argument("--weights_dir", type=str, help="The path to the weights of the networks", default=os.path.join(os.path.dirname(__file__), "weights"))
    parser.add_argument("--csv_file", type=str, help="The path to the csv file", default=os.path.join(os.path.dirname(__file__), "operations.csv"))
    args = vars(parser.parse_args())
    runnig_tests(args['data_dir'], args['out_dir'], args['weights_dir'], args['csv_file'])

main()
evaluate_results(os.path.join(os.path.dirname(__file__), "results_test"), ['Corvi_pretrain'])
