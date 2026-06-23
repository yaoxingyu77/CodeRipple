import csv
import torch
import numpy as np
import pywt
from data_process import llama_entropy
from tqdm import tqdm
from scipy.stats import kurtosis
from datasets import load_from_disk, Dataset
from scipy import signal
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pyentrp import entropy as ent
from sklearn.preprocessing import MinMaxScaler

le = llama_entropy()

def compute_sampen(series, m=3, r=0.2):
    std = np.std(series)
    if std == 0:
        return 0
    sampens = ent.sample_entropy(series, m, r * std)
    return sampens[-1]  

def add_sampen_to_dataset(dataset, num_samples):
    swt_wavlet_feature = dataset['swt_wavelet_feature']

    
    data = {
        "CD1": [],
        "CD2": [],
        "CD3": [],
        "CD4": [],
        "CA4": [],
    }
    valid_indices = []  

    for i in range(num_samples):
        sample = swt_wavlet_feature[i]
        
        if sample is None or len(sample)==0:  
            print(f"skip index {i}")
            continue
        valid_indices.append(i)
        for j, key in enumerate(["CD1", "CD2", "CD3", "CD4", "CA4"]):
            data[key].append(sample[j])
  
    
    normalized_data = {}
    for key in data:
        stacked = np.stack(data[key]) 
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(stacked)
        normalized_data[key] = normalized
    
    sampen_results = {f"SampEn_{key}": [None] * num_samples for key in ["CD1","CD2","CD3","CD4","CA4"]}

   
    for idx, i in enumerate(valid_indices):
        for key in ["CD1", "CD2", "CD3", "CD4", "CA4"]:
            sampen_value = compute_sampen(normalized_data[key][idx])
            sampen_results[f"SampEn_{key}"][i] = sampen_value
            

    for key in sampen_results:
        dataset = dataset.add_column(key, sampen_results[key])

    return dataset

def pad_to_length_256(signal) -> torch.Tensor:
    if isinstance(signal, list):
        signal = torch.tensor(signal, dtype=torch.float32)
    elif isinstance(signal, np.ndarray):
        signal = torch.tensor(signal, dtype=torch.float32)
    elif not isinstance(signal, torch.Tensor):
        raise TypeError("Input must be list, np.ndarray or torch.Tensor")

    if signal.shape[0] < 256:
        signal = torch.cat([signal, torch.zeros(256 - signal.shape[0], dtype=signal.dtype)])
    else:
        signal = signal[:256]
    return signal

def stationary_wavelet_transform(signal, wavelet='db6', level=4):
    
    if isinstance(signal, torch.Tensor):
        signal_np = signal.detach().cpu().numpy()
    elif isinstance(signal, np.ndarray):
        signal_np = signal
    elif isinstance(signal, list):
        signal_np = np.array(signal)
    else:
        raise TypeError(f"Unsupported type: {type(signal)}")
    coeffs = pywt.swt(signal_np, wavelet=wavelet, level=level, trim_approx=False)
    results = {}
    for i, (cA, cD) in enumerate(coeffs):
        real_level = level - i
        results[f'cA{real_level}'] = torch.tensor(cA, dtype=torch.float32)
        results[f'cD{real_level}'] = torch.tensor(cD, dtype=torch.float32)
    return results

def stack_swt_selected_features(swt_dict):
    channels = [swt_dict[f'cD{i}'] for i in range(1, 5)]
    channels.append(swt_dict[f'cA4'])
    return torch.stack(channels, dim=0)

def compute_detail_kurtosis(signal, wavelet='db6', level=4):
    
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cDs = coeffs[1:]  
    all_details = np.concatenate(cDs)
    k = kurtosis(all_details, fisher=False, bias=False)
    return k


def zero_shot_eval(features, labels):

    auroc = roc_auc_score(labels, features)
    return {
        "AUROC": auroc    
    }



def process_csv(csv_path, output_dir):
    all_samples = []
    sample_id = 0 
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"Processing {os.path.basename(csv_path)}"):
            entries = [
                (row.get("Python Code", ""), 0),
                (row.get("GPT Answer", ""), 1)
            ]
            try:
                ppls = le.compute_score([e[0] for e in entries])
            except Exception as e:
                print(f"PPL compute error: {e}")
                continue


            for (code, label), ppl in zip(entries, ppls):
                sample_id += 1
                if ppl and len(ppl)>0:
                    try:
            
                        ppl_512_tensor = ppl
                        ppl_tensor = pad_to_length_256(ppl).tolist()
                        dwt_K_feature = compute_detail_kurtosis(ppl_tensor, wavelet='db6')
                    except Exception as e:
                        print(f"PPL padding error on sample {sample_id}: {e}")
                        ppl_tensor = []
                        ppl_512_tensor = []
                        dwt_K_feature = np.nan
                else:
                    ppl_tensor = []
                    ppl_512_tensor = []
                    dwt_K_feature = np.nan

                try:
                    all_samples.append({
                        "id": sample_id,
                        "label": label,
                        "code": code,
                        "ppl_512": ppl_512_tensor,
                        "ppl": ppl_tensor,
                        "dwt_K_feature": dwt_K_feature
                    })
                except Exception as e:
                    print(f"Wavelet error: {e}")
                    continue

    if all_samples:
        hf_dataset = Dataset.from_list(all_samples)
        hf_dataset.save_to_disk(output_dir)
        print(f"Saved {output_dir}, total {len(all_samples)} samples")
    else:
        print(f"No valid samples in {csv_path}")

def add_swt_features(example):
    ppl = example["ppl"]
    if ppl is None or len(ppl) == 0:
        swt_feature = []
    else:
        try:
            results = stationary_wavelet_transform(ppl)
            swt_feature = stack_swt_selected_features(results)
        except Exception as e:
            print(f"Error computing swt_feature for id={example.get('id', 'unknown')}: {e}")
            swt_feature = []
    example["swt_wavelet_feature"] = swt_feature
    return example


if __name__ == "__main__":
    base_dir = "./AI_Code_Detection_Education" # Path of the dataset.
    for i in range(4,14):  
        csv_filename = f"variant_{i}_full.csv"
        csv_path = os.path.join(base_dir, csv_filename)
        
        print(f"Processing {csv_path} ...")
        output_dir = f'./CodeRippleEdu/edu{i}' # save path
        os.makedirs(output_dir, exist_ok=True)
        process_csv(csv_path, output_dir)
    
      
        dataset = load_from_disk(output_dir)
        dataset = dataset.map(add_swt_features)
        num_samples = len(dataset)
        sampen_dataset = add_sampen_to_dataset(dataset, num_samples)
        output_dir1 = output_dir+'_swt_db6_sampen'
        os.makedirs(output_dir1, exist_ok=True)
        sampen_dataset.save_to_disk(output_dir1)
        print("Saved！")

        keys = ["SampEn_CD1", "SampEn_CD2", "SampEn_CD3", "SampEn_CD4", "SampEn_CA4"]
        kurtosis_vals = np.array(sampen_dataset["dwt_K_feature"])
        labels = np.array(sampen_dataset["label"])
        ppl = np.array(sampen_dataset["ppl"])
        sampen_cols = {key: np.array(sampen_dataset[key]) for key in keys}
 
        mask = (
            (kurtosis_vals != None) &
            (~np.isnan(kurtosis_vals)) &
            (labels != None) &
            np.array([p is not None and len(p) > 0 for p in ppl], dtype=bool)
        )
        for key in keys:
            mask &= np.array([v is not None for v in sampen_cols[key]])

        valid_indices = np.where(mask)[0]
        print(f'variant {i}')
        print(f"valid count = {len(valid_indices)}") 

       
        invalid_indices = np.where(~mask)[0]
        for idx in invalid_indices:
            reasons = []
            if kurtosis_vals[idx] is None:
                reasons.append("kurtosis is None")
            elif isinstance(kurtosis_vals[idx], (float, int)) and np.isnan(kurtosis_vals[idx]):
                reasons.append("kurtosis is NaN")
            if labels[idx] is None:
                reasons.append("label is None")
            if ppl[idx] is None or len(ppl[idx]) == 0:
                reasons.append("ppl empty")
            for key in keys:
                if sampen_cols[key][idx] is None:
                    reasons.append(f"{key} is None")
            print(f"idx {idx}: {', '.join(reasons)}")

       
        kurtosis_vals = kurtosis_vals[valid_indices]
        sampen_features = np.column_stack([sampen_cols[key][valid_indices] for key in keys])
        labels = labels[valid_indices]


    

       
        features = np.hstack([kurtosis_vals.reshape(-1, 1), sampen_features])

        
        L = [0.5, 0.4, 0.3, 0.2, 0.1, 0]
        x = 0.01  
        factor = (1 - x) / sum(L[1:])
        new_L = [x] + [v * factor for v in L[1:]]
        print(new_L)
        weights = np.array(new_L)
        scores = features.dot(weights)
        result = zero_shot_eval(scores, labels)
        print(result)
        