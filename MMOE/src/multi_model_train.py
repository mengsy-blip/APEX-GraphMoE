


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import gc
import time
import h5py
import random




def load_single_h5_file_to_memory_full_embeddings(h5_file, max_samples=None):
    print(f" textfile: {os.path.basename(h5_file)}")

    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"file does not exist: {h5_file}")

    try:
        with h5py.File(h5_file, 'r') as f:

            print(f"   textdimension consistency...")

            sample_counts = []
            dataset_info = {}


            all_datasets = list(f.keys())

            for dataset_name in all_datasets:
                if isinstance(f[dataset_name], h5py.Dataset) and len(f[dataset_name].shape) > 0:
                    sample_count = f[dataset_name].shape[0]
                    sample_counts.append(sample_count)
                    dataset_info[dataset_name] = sample_count


            if len(set(sample_counts)) > 1:
                min_samples = min(sample_counts)
                max_samples_found = max(sample_counts)

                print(f"   texttodimension mismatch issue:")
                print(f"    textnumber of samples: {min_samples:,}")
                print(f"    textnumber of samples: {max_samples_found:,}")
                print(f"    text: {max_samples_found - min_samples:,} text")
                print(f"   auto fix: truncated totext {min_samples:,}")


                num_samples = min_samples
            else:
                num_samples = sample_counts[0] if sample_counts else 0
                print(f"   dimensions are consistent: {num_samples:,} text")


            if max_samples and num_samples > max_samples:
                num_samples = max_samples
                print(f"   usertext: truncated to {num_samples:,} text")

            print(f"   final sample count: {num_samples:,}")


            if 'image_embeddings' in f:
                image_emb = f['image_embeddings'][:num_samples]
                print(f"    image embeddings: {image_emb.shape}")
            else:
                image_emb = np.zeros((num_samples, 256), dtype=np.float32)
                print(f"    image embeddings: using zero vectors {image_emb.shape}")


            text_fields = ['item_title', 'item_entity_names', 'bill_entity_seq', 'service_entity_seq', 'query_entity_seq']
            text_embeddings_list = []

            for field in text_fields:
                field_key = f'{field}_embeddings'
                if field_key in f:
                    field_emb = f[field_key][:num_samples]
                    text_embeddings_list.append(field_emb)
                    print(f"    {field}: {field_emb.shape}")
                else:
                    field_emb = np.zeros((num_samples, 768), dtype=np.float32)
                    text_embeddings_list.append(field_emb)
                    print(f"    {field}: using zero vectors {field_emb.shape}")

            text_emb = np.concatenate(text_embeddings_list, axis=1)
            print(f"    text embeddingstext: {text_emb.shape}")


            print(f"   reading ID embeddings...")
            id_embeddings_list = []


            id_feature_names = [
                'user_id_embeddings',
                'item_id_embeddings',
                'deep_features_0_embeddings',
                'deep_features_1_embeddings',
                'deep_features_2_embeddings',
                'deep_features_3_embeddings',
                'deep_features_4_embeddings',
                'deep_features_5_embeddings',
                'deep_features_6_embeddings',
                'deep_features_7_embeddings',
                'deep_features_8_embeddings',
                'deep_features_9_embeddings',
                'deep_features_10_embeddings',
                'deep_features_11_embeddings',
                'deep_features_12_embeddings',
                'deep_features_13_embeddings',
                'deep_features_14_embeddings',
                'deep_features_15_embeddings',
                'deep_features_16_embeddings',
                'deep_features_17_embeddings',
                'deep_features_18_embeddings',
                'deep_features_19_embeddings',
                'deep_features_21_embeddings',
                'deep_features_22_embeddings',
                'deep_features_23_embeddings',
                'deep_features_24_embeddings',
                'deep_features_25_embeddings',
                'deep_features_26_embeddings'
            ]

            print(f"    expected number of ID features: {len(id_feature_names)}")

            for feature_name in id_feature_names:
                if feature_name in f:
                    field_emb = f[feature_name][:num_samples]
                    id_embeddings_list.append(field_emb)

                    if len(id_embeddings_list) <= 5 or 'deep_features_26' in feature_name:
                        print(f"    {feature_name}: {field_emb.shape}")
                else:

                    field_emb = np.zeros((num_samples, 32), dtype=np.float32)
                    id_embeddings_list.append(field_emb)
                    print(f"    {feature_name}: missing,using zero vectors {field_emb.shape}")

            print(f"    actual number of ID features read: {len(id_embeddings_list)}")


            id_emb = np.concatenate(id_embeddings_list, axis=1)
            print(f"     ID embeddingstext: {id_emb.shape}")


            labels = f['labels'][:num_samples]
            print(f"    labels: {labels.shape}")

            metadata = f['metadata'][:num_samples]
            print(f"    metadata: {metadata.shape}")


            scenes = []
            user_ids = []
            item_ids = []

            print(f"    processing metadata...")
            for j in range(num_samples):
                meta = metadata[j]


                user_id_str = meta[0].decode('utf-8') if isinstance(meta[0], bytes) else str(meta[0])
                user_ids.append(user_id_str)


                item_id_str = meta[1].decode('utf-8') if isinstance(meta[1], bytes) else str(meta[1])
                item_ids.append(item_id_str)


                scene_str = meta[2].decode('utf-8') if isinstance(meta[2], bytes) else str(meta[2])
                scene_id = int(scene_str) if scene_str.isdigit() else 0
                scenes.append(scene_id)


            single_file_data = {
                'image_embeddings': image_emb,
                'text_embeddings': text_emb,
                'id_embeddings': id_emb,
                'labels': labels,
                'scenes': np.array(scenes, dtype=np.int64),
                'user_ids': user_ids,
                'item_ids': item_ids
            }


            print(f"   textdata consistency:")
            for key, value in single_file_data.items():
                if key in ['user_ids', 'item_ids']:
                    print(f"    {key}: {len(value)} ")
                else:
                    print(f"    {key}: {value.shape}")


            shapes = []
            for key, value in single_file_data.items():
                if key in ['user_ids', 'item_ids']:
                    shapes.append(len(value))
                else:
                    shapes.append(value.shape[0])

            if len(set(shapes)) == 1:
                print(f"   data-consistency validationtext: allfeaturestext {shapes[0]:,} text")
            else:
                print(f"   data-consistency validationfailed: {set(shapes)}")
                raise ValueError("inconsistent data!")

            total_positive = np.sum(labels)
            print(f"   positive samples: {int(total_positive):,} ({total_positive/num_samples:.2%})")

            return single_file_data

    except Exception as e:
        print(f" textfiletext: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_memory_dataset_full_embeddings(data_dict):
    print(" textPyTorchdataset...")


    tensors = []
    tensor_names = [
        'image_embeddings', 'text_embeddings', 'id_embeddings',
        'labels', 'scenes'
    ]

    for name in tensor_names:
        if name in data_dict:
            if name == 'labels':
                tensor = torch.from_numpy(data_dict[name]).float()
            elif name == 'scenes':
                tensor = torch.from_numpy(data_dict[name]).long()
            else:
                tensor = torch.from_numpy(data_dict[name]).float()
            tensors.append(tensor)
            print(f"  {name}: {tensor.shape}")
        else:
            print(f"   missing data: {name}")


    user_ids = data_dict.get('user_ids', [])
    item_ids = data_dict.get('item_ids', [])


    class StreamingDataset:
        def __init__(self, tensors, user_ids, item_ids):
            self.tensors = tensors
            self.user_ids = user_ids
            self.item_ids = item_ids
            self.length = len(tensors[0]) if tensors else 0

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            tensor_data = [tensor[idx] for tensor in self.tensors]
            user_id = self.user_ids[idx] if idx < len(self.user_ids) else "default_user"
            item_id = self.item_ids[idx] if idx < len(self.item_ids) else "default_item"
            return tuple(tensor_data + [user_id, item_id])

    dataset = StreamingDataset(tensors, user_ids, item_ids)
    print(f" dataset creationcompleted: {len(dataset):,} text(supportsLightGCN graph expert)")

    return dataset


def calculate_scene_auc(y_true, y_pred, scenes):
    results = {}
    for i in range(5):
        scene_indices = [idx for idx, s in enumerate(scenes) if s == i]
        if len(scene_indices) == 0:
            results[f"scene_{i}_auc"] = None
            continue

        scene_y_true = [y_true[idx] for idx in scene_indices]
        scene_y_pred = [y_pred[idx] for idx in scene_indices]

        unique_labels = set(scene_y_true)
        if len(unique_labels) > 1 and len(scene_y_true) > 1:
            try:
                results[f"scene_{i}_auc"] = roc_auc_score(scene_y_true, scene_y_pred)
            except Exception:
                results[f"scene_{i}_auc"] = None
        else:
            results[f"scene_{i}_auc"] = None
    return results




def train_epoch_streaming_full_embeddings(model, train_files, optimizer, loss_function, device,
                                         batch_size=1024, max_samples_per_file=None, shuffle_files=True):
    model.train()
    total_loss = 0
    total_diversity_loss = 0
    y_true_list = []
    y_pred_list = []
    scene_list = []
    total_batches = 0
    total_samples = 0


    files_to_process = train_files.copy()
    if shuffle_files:
        random.shuffle(files_to_process)

    print(f" Top-Kanti-polarizationtexttraining: text {len(files_to_process)} file")

    for file_idx, train_file in enumerate(files_to_process):
            print(f"\n processing training file [{file_idx+1}/{len(files_to_process)}]: {os.path.basename(train_file)}")

        # try:

            single_file_data = load_single_h5_file_to_memory_full_embeddings(
                train_file, max_samples=max_samples_per_file
            )


            train_dataset = create_memory_dataset_full_embeddings(single_file_data)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True
            )

            file_batches = len(train_loader)
            file_samples = len(train_dataset)
            print(f"  number of batches: {file_batches}, number of samples: {file_samples:,}")


            pbar = tqdm(train_loader, desc=f"training files{file_idx+1}")

            for batch in pbar:

                image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch


                image_emb = image_emb.to(device, non_blocking=True)
                text_emb = text_emb.to(device, non_blocking=True)
                id_emb = id_emb.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                scenes = scenes.to(device, non_blocking=True)


                optimizer.zero_grad()


                # if hasattr(model, 'forward') and 'diversity_loss' in str(model.forward):


                predictions, diversity_loss = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)

                # else:


                #     predictions = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
                #     diversity_loss = torch.tensor(0.0, device=device)


                main_loss = loss_function(predictions, labels)


                diversity_weight = getattr(model, 'diversity_weight', 0.01)
                total_loss_batch = main_loss + diversity_weight * diversity_loss


                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()


                total_loss += main_loss.item()
                total_diversity_loss += diversity_loss.item()
                total_batches += 1


                with torch.no_grad():
                    y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
                    y_true = labels.detach().cpu().numpy()
                    scene_ids = scenes.detach().cpu().numpy()

                    y_true_list.extend(y_true.tolist())
                    y_pred_list.extend(y_pred.tolist())
                    scene_list.extend(scene_ids.tolist())


                pbar.set_postfix({
                    'Loss': f'{main_loss.item():.4f}',
                    'Div': f'{diversity_loss.item():.5f}'
                })

            total_samples += file_samples


            del single_file_data, train_dataset, train_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # except Exception as e:

        #     continue


    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_diversity_loss = total_diversity_loss / total_batches if total_batches > 0 else 0

    overall_auc = 0
    scene_aucs = {}
    try:
        if len(set(y_true_list)) > 1 and len(y_true_list) > 0:
            overall_auc = roc_auc_score(y_true_list, y_pred_list)
            scene_aucs = calculate_scene_auc(y_true_list, y_pred_list, scene_list)
            print(f"   Top-Kanti-polarizationtrainingAUCtextcompleted,based on {len(y_true_list):,} text")
    except Exception as e:
        print(f" AUCcalculation error: {e}")

    return avg_loss, overall_auc, scene_aucs, total_samples, avg_diversity_loss


def evaluate_epoch_streaming_full_embeddings(model, test_files, loss_function, device,
                                            batch_size=1024, max_samples_per_file=None):
    model.eval()
    total_loss = 0
    total_diversity_loss = 0
    y_true_list = []
    y_pred_list = []
    scene_list = []
    total_batches = 0
    total_samples = 0

    print(f" Top-Kanti-polarizationtextevaluation: text {len(test_files)} test files")

    with torch.no_grad():
        for file_idx, test_file in enumerate(test_files):
            print(f"\n processing test file [{file_idx+1}/{len(test_files)}]: {os.path.basename(test_file)}")

            try:

                single_file_data = load_single_h5_file_to_memory_full_embeddings(
                    test_file, max_samples=max_samples_per_file
                )


                test_dataset = create_memory_dataset_full_embeddings(single_file_data)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True if torch.cuda.is_available() else False,
                    drop_last=False
                )

                file_batches = len(test_loader)
                file_samples = len(test_dataset)
                print(f"  number of batches: {file_batches}, number of samples: {file_samples:,}")


                pbar = tqdm(test_loader, desc=f"evaluationfile{file_idx+1}")

                for batch in pbar:

                    image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch


                    image_emb = image_emb.to(device, non_blocking=True)
                    text_emb = text_emb.to(device, non_blocking=True)
                    id_emb = id_emb.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    scenes = scenes.to(device, non_blocking=True)



                    # if hasattr(model, 'forward') and 'diversity_loss' in str(model.forward):

                    predictions, diversity_loss = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
                    # else:

                        # predictions = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
                        # diversity_loss = torch.tensor(0.0, device=device)


                    main_loss = loss_function(predictions, labels)
                    total_loss += main_loss.item()
                    total_diversity_loss += diversity_loss.item()
                    total_batches += 1


                    y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
                    y_true = labels.detach().cpu().numpy()
                    scene_ids = scenes.detach().cpu().numpy()

                    y_true_list.extend(y_true.tolist())
                    y_pred_list.extend(y_pred.tolist())
                    scene_list.extend(scene_ids.tolist())


                    pbar.set_postfix({
                        'Loss': f'{main_loss.item():.4f}',
                        'Div': f'{diversity_loss.item():.5f}'
                    })

                total_samples += file_samples


                del single_file_data, test_dataset, test_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"   processing file {test_file}  failed: {e}")
                continue


    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_diversity_loss = total_diversity_loss / total_batches if total_batches > 0 else 0

    overall_auc = 0
    scene_aucs = {}
    try:
        if len(set(y_true_list)) > 1 and len(y_true_list) > 0:
            overall_auc = roc_auc_score(y_true_list, y_pred_list)
            scene_aucs = calculate_scene_auc(y_true_list, y_pred_list, scene_list)
            print(f"   Top-Kanti-polarizationtestingAUCtextcompleted,based on {len(y_true_list):,} text")
    except Exception as e:
        print(f" AUCcalculation error: {e}")

    return avg_loss, overall_auc, scene_aucs, total_samples, avg_diversity_loss




def load_all_h5_data_to_memory_full_embeddings(h5_files, max_samples=None, use_mmap=True):
    print(" compatibility interface:usingtext...")

    if not h5_files:
        raise ValueError("file list is empty!")


    first_file = h5_files[0]
    print(f" compatibility mode:textfile {os.path.basename(first_file)}")

    return load_single_h5_file_to_memory_full_embeddings(first_file, max_samples)


def train_epoch_memory_full_embeddings(model, train_loader_or_files, optimizer, loss_function, device):

    if isinstance(train_loader_or_files, list):
        result = train_epoch_streaming_full_embeddings(
            model, train_loader_or_files, optimizer, loss_function, device
        )

        if len(result) == 5:
            return result[:4]
        return result


    print(" usingDataLoadertext(supportsLightGCN graph expert+Top-Kanti-polarization)")
    train_loader = train_loader_or_files

    model.train()
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    scene_list = []
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")

    for batch in pbar:

        if len(batch) == 7:
            image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch
        else:
            image_emb, text_emb, id_emb, labels, scenes = batch[:5]
            user_ids, item_ids = None, None


        image_emb = image_emb.to(device, non_blocking=True)
        text_emb = text_emb.to(device, non_blocking=True)
        id_emb = id_emb.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        scenes = scenes.to(device, non_blocking=True)


        optimizer.zero_grad()


        if hasattr(model, 'forward') and 'diversity_loss' in str(model.forward):

            predictions, diversity_loss = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
            main_loss = loss_function(predictions, labels)
            diversity_weight = getattr(model, 'diversity_weight', 0.01)
            total_loss_batch = main_loss + diversity_weight * diversity_loss
        else:

            predictions = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
            total_loss_batch = loss_function(predictions, labels)


        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        total_loss += total_loss_batch.item()
        num_batches += 1


        with torch.no_grad():
            y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            scene_ids = scenes.detach().cpu().numpy()

            y_true_list.extend(y_true.tolist())
            y_pred_list.extend(y_pred.tolist())
            scene_list.extend(scene_ids.tolist())

        pbar.set_postfix({'Loss': f'{total_loss_batch.item():.4f}'})


    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    overall_auc = 0
    scene_aucs = {}
    try:
        if len(set(y_true_list)) > 1:
            overall_auc = roc_auc_score(y_true_list, y_pred_list)
            scene_aucs = calculate_scene_auc(y_true_list, y_pred_list, scene_list)
    except Exception as e:
        print(f" AUCcalculation error: {e}")

    return avg_loss, overall_auc, scene_aucs, len(y_true_list)


def evaluate_epoch_memory_full_embeddings(model, test_loader_or_files, loss_function, device):

    if isinstance(test_loader_or_files, list):
        result = evaluate_epoch_streaming_full_embeddings(
            model, test_loader_or_files, loss_function, device
        )

        if len(result) == 5:
            return result[:4]
        return result


    print(" usingDataLoadertext(supportsLightGCN graph expert+Top-Kanti-polarization)")
    test_loader = test_loader_or_files

    model.eval()
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    scene_list = []
    num_batches = 0

    pbar = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad():
        for batch in pbar:

            if len(batch) == 7:
                image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch
            else:
                image_emb, text_emb, id_emb, labels, scenes = batch[:5]
                user_ids, item_ids = None, None


            image_emb = image_emb.to(device, non_blocking=True)
            text_emb = text_emb.to(device, non_blocking=True)
            id_emb = id_emb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            scenes = scenes.to(device, non_blocking=True)



            if hasattr(model, 'forward') and 'diversity_loss' in str(model.forward):

                predictions, diversity_loss = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
            else:

                predictions = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)


            loss = loss_function(predictions, labels)
            total_loss += loss.item()
            num_batches += 1


            y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            scene_ids = scenes.detach().cpu().numpy()

            y_true_list.extend(y_true.tolist())
            y_pred_list.extend(y_pred.tolist())
            scene_list.extend(scene_ids.tolist())

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})


    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    overall_auc = 0
    scene_aucs = {}
    try:
        if len(set(y_true_list)) > 1:
            overall_auc = roc_auc_score(y_true_list, y_pred_list)
            scene_aucs = calculate_scene_auc(y_true_list, y_pred_list, scene_list)
    except Exception as e:
        print(f" AUCcalculation error: {e}")

    return avg_loss, overall_auc, scene_aucs, len(y_true_list)


def print_memory_usage(stage=""):
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"   {stage} GPUtext: {gpu_memory:.2f}GB text / {gpu_cached:.2f}GB text")

    try:
        import psutil
        cpu_memory = psutil.virtual_memory()
        cpu_used_gb = cpu_memory.used / 1024**3
        cpu_total_gb = cpu_memory.total / 1024**3
        print(f"   {stage} CPUtext: {cpu_used_gb:.2f}GB / {cpu_total_gb:.2f}GB ({cpu_memory.percent:.1f}%)")
    except ImportError:
        print(f"   {stage} unable to getCPUtext(requires installingpsutil)")

def create_expert_analysis_dataloader(test_files, device, batch_size=1024, max_samples_per_file=None, max_files_for_analysis=2):
    print(f" forexpert weightsanalysispreparingdata loadingtext...")


    analysis_files = test_files[:max_files_for_analysis]
    print(f"  using {len(analysis_files)} test filesrunningexpert analysis")


    if analysis_files:
        analysis_file = analysis_files[0]
        print(f"   analysis file: {os.path.basename(analysis_file)}")

        try:

            single_file_data = load_single_h5_file_to_memory_full_embeddings(
                analysis_file, max_samples=max_samples_per_file
            )


            analysis_dataset = create_memory_dataset_full_embeddings(single_file_data)


            class DeviceSafeDataLoader:
                def __init__(self, dataset, batch_size, device):
                    self.dataset = dataset
                    self.batch_size = batch_size
                    self.device = device
                    self.length = len(dataset)

                def __len__(self):
                    return (self.length + self.batch_size - 1) // self.batch_size

                def __iter__(self):
                    for i in range(0, self.length, self.batch_size):
                        end_idx = min(i + self.batch_size, self.length)
                        batch_data = []
                        batch_user_ids = []
                        batch_item_ids = []


                        for j in range(i, end_idx):
                            data = self.dataset[j]
                            # data = [image, text, id_emb, label, scene, user_id, item_id]
                            batch_data.append(data[:5])
                            batch_user_ids.append(data[5])  # user_id
                            batch_item_ids.append(data[6])  # item_id


                        if batch_data:
                            tensors = []
                            for tensor_idx in range(5):  # image, text, id_emb, label, scene
                                tensor_list = [item[tensor_idx] for item in batch_data]
                                stacked_tensor = torch.stack(tensor_list).to(self.device, non_blocking=True)
                                tensors.append(stacked_tensor)


                            yield (*tensors, batch_user_ids, batch_item_ids)


            analysis_loader = DeviceSafeDataLoader(analysis_dataset, batch_size, device)

            print(f"   expert analysisdata loadingtextcompleted: {len(analysis_dataset):,} text (devicetext)")
            return analysis_loader

        except Exception as e:
            print(f"   textexpert analysisdata loadingtextfailed: {e}")
            import traceback
            traceback.print_exc()
            return None

    return None


def train_model_memory_single_gpu_full_embeddings(model_class, model_kwargs, train_files, test_files,
                                                 epochs=50, lr=1e-4, batch_size=512, save_path="model/best_model.pt",
                                                 early_stop=5, max_samples=None, use_mmap=True,
                                                 enable_expert_analysis=True, expert_analysis_interval=1):
    print(f" startingTop-Kanti-polarizationtextGPUtraining(full-embedding mode+LightGCN graph expert+Top-Kanti-polarization+expert-importance analysis,memory optimizationtext)...")
    print_memory_usage("text")


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f" using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print(" using CPU for training")

    print(f" Top-Kanti-polarizationtexttraining configuration:")
    print(f"  training mode: file-by-file streaming(memory optimized)")
    print(f"  training files: {len(train_files)} ")
    print(f"  test files: {len(test_files)} ")
    print(f"  batch size: {batch_size}")
    print(f"  per-file sample limit: {max_samples or 'unlimited'}")
    print(f"   supportsLightGCN graph expert(userIDanditemID)")
    print(f"   Top-Kanti-polarization: textphasedynamictraining strategy")
    print(f"   expert-importance analysis: {'enabled' if enable_expert_analysis else 'disabled'}")
    if enable_expert_analysis:
        print(f"   analysis interval: every {expert_analysis_interval} epoch")


    print("\n textTop-Kanti-polarizationtext...")
    model = model_class(**model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" model parameters:")
    print(f"  total parameters: {total_params:,}")
    print(f"  trainable parameters: {trainable_params:,}")
    if hasattr(model, 'use_lightgcn'):
        print(f"   LightGCN graph expert: {'enabled' if model.use_lightgcn else 'disabled'}")
    if hasattr(model, 'topk_config'):
        print(f"   Top-Kanti-polarization: enabled(dynamicKtextstrategy)")


    has_expert_analysis = hasattr(model, 'print_expert_weights_after_training')
    if enable_expert_analysis and has_expert_analysis:
        print(f"   expert-importance analysis: enabled(supportsTop-Kanti-polarizationanalysis)")
    elif enable_expert_analysis and not has_expert_analysis:
        print(f"   expert-importance analysis: model does not support this,disabled")
        enable_expert_analysis = False


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_function = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    best_auc = 0
    patience = 0
    train_history = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n startingTop-Kanti-polarizationtexttraining loop:")
    print(f"  device: {device}")
    print(f"  learning rate: {lr}")
    print(f"  training epochs: {epochs}")
    print(f"  early-stopping patience: {early_stop}")
    print(f"   LightGCN graph expert: {'enabled' if model_kwargs.get('use_lightgcn', False) else 'disabled'}")
    print(f"   Top-Kanti-polarization: {'enabled' if hasattr(model, 'topk_config') else 'not detected'}")
    print(f"   expert-importance analysis: {'enabled' if enable_expert_analysis else 'not enabled'}")
    print("=" * 60)


    for epoch in range(epochs):
        start_time = time.time()
        print_memory_usage(f"Epoch {epoch+1} starting")


        if hasattr(model, 'update_epoch'):
            model.update_epoch(epoch)

            if hasattr(model, 'get_training_phase_info'):
                phase_info = model.get_training_phase_info()
                print(f"\n Top-Ktrainingphase [{epoch+1}]: {phase_info['training_phase']}")
                params = phase_info['current_topk_params']
                print(f"  parameters: K={params['k']}, max_weight={params['max_weight']:.2f}, "
                      f"temp={params['temperature']:.1f}, alpha={params['alpha']:.2f}")
                print(f"  strategy: {phase_info['phase_description']}")


        train_result = train_epoch_streaming_full_embeddings(
            model, train_files, optimizer, loss_function, device,
            batch_size=batch_size, max_samples_per_file=max_samples
        )


        if len(train_result) == 5:
            train_loss, train_auc, train_scene_aucs, train_samples, train_diversity_loss = train_result
        else:
            train_loss, train_auc, train_scene_aucs, train_samples = train_result
            train_diversity_loss = 0.0

        print_memory_usage(f"Epoch {epoch+1} after training")


        test_result = evaluate_epoch_streaming_full_embeddings(
            model, test_files, loss_function, device,
            batch_size=batch_size, max_samples_per_file=max_samples
        )


        if len(test_result) == 5:
            test_loss, test_auc, test_scene_aucs, test_samples, test_diversity_loss = test_result
        else:
            test_loss, test_auc, test_scene_aucs, test_samples = test_result
            test_diversity_loss = 0.0

        print_memory_usage(f"Epoch {epoch+1} after testing")


        scheduler.step()

        epoch_time = time.time() - start_time

        print(f"\n Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
        print(f"  training - Loss: {train_loss:.4f}, AUC: {train_auc:.4f} (text: {train_samples:,})")
        print(f"  testing - Loss: {test_loss:.4f}, AUC: {test_auc:.4f} (text: {test_samples:,})")
        print(f"  learning rate: {scheduler.get_last_lr()[0]:.2e}")


        if train_diversity_loss > 0 or test_diversity_loss > 0:
            print(f"   text - training: {train_diversity_loss:.5f}, testing: {test_diversity_loss:.5f}")


        if train_scene_aucs:
            scene_auc_str = ", ".join([f"S{i}: {v:.3f}" if v else f"S{i}: N/A"
                                     for i, v in enumerate([train_scene_aucs.get(f"scene_{j}_auc") for j in range(5)])])
            print(f"  trainingsceneAUC - {scene_auc_str}")

        if test_scene_aucs:
            scene_auc_str = ", ".join([f"S{i}: {v:.3f}" if v else f"S{i}: N/A"
                                     for i, v in enumerate([test_scene_aucs.get(f"scene_{j}_auc") for j in range(5)])])
            print(f"  testingsceneAUC - {scene_auc_str}")


        if enable_expert_analysis and has_expert_analysis and (epoch + 1) % expert_analysis_interval == 0:
            print(f"\n runningexpert-importance analysis...")
            try:

                analysis_loader = create_expert_analysis_dataloader(
                    test_files, device, batch_size=batch_size,
                    max_samples_per_file=max_samples, max_files_for_analysis=1
                )

                if analysis_loader is not None:

                    model.print_expert_weights_after_training(
                        data_loader=analysis_loader,
                        epoch=epoch + 1,
                        max_batches=3
                    )


                    del analysis_loader
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print(f"   textexpert analysisdata loadingtext,textanalysis")

            except Exception as e:
                print(f"   expert-importance analysisfailed: {e}")



        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'test_loss': test_loss,
            'test_auc': test_auc,
            'lr': scheduler.get_last_lr()[0],
            'train_samples': train_samples,
            'test_samples': test_samples,
            'train_diversity_loss': train_diversity_loss,
            'test_diversity_loss': test_diversity_loss
        }


        for i in range(5):
            key = f"scene_{i}_auc"
            history_entry[f'train_{key}'] = train_scene_aucs.get(key) if train_scene_aucs else None
            history_entry[f'test_{key}'] = test_scene_aucs.get(key) if test_scene_aucs else None


        if hasattr(model, 'get_training_phase_info'):
            phase_info = model.get_training_phase_info()
            topk_params = phase_info['current_topk_params']
            history_entry.update({
                'topk_phase': phase_info['training_phase'],
                'topk_k': topk_params['k'],
                'topk_max_weight': topk_params['max_weight'],
                'topk_temperature': topk_params['temperature'],
                'topk_alpha': topk_params['alpha']
            })

        train_history.append(history_entry)


        if test_auc > best_auc:
            best_auc = test_auc
            patience = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_auc': train_auc,
                'test_loss': test_loss,
                'test_auc': test_auc,
                'train_history': train_history,
                'model_kwargs': model_kwargs,
                'feature_mode': 'topk_full_embeddings_streaming_with_lightgcn_expert_analysis',
                'use_streaming': True,
                'use_lightgcn': model_kwargs.get('use_lightgcn', False),
                'use_topk_antipolarization': hasattr(model, 'topk_config'),
                'topk_config': getattr(model, 'topk_config', None) if hasattr(model, 'topk_config') else None,
                'expert_analysis_enabled': enable_expert_analysis
            }

            torch.save(checkpoint, save_path)
            print(f"   saving bestTop-Kanti-polarizationtext (AUC: {best_auc:.4f})")

        else:
            patience += 1
            print(f"   patience: {patience}/{early_stop}")


        if patience >= early_stop:
            print(f"   early stopping triggered!")
            break


        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    if enable_expert_analysis and has_expert_analysis:
        print(f"\n training completed,runningtextexpert-importance analysis...")
        try:

            analysis_loader = create_expert_analysis_dataloader(
                test_files, device, batch_size=batch_size,
                max_samples_per_file=max_samples, max_files_for_analysis=2
            )

            if analysis_loader is not None:

                model.print_expert_weights_after_training(
                    data_loader=analysis_loader,
                    epoch="Final",
                    max_batches=5
                )


                del analysis_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"   textexpert analysisdata loadingtext,textfinal analysis")

        except Exception as e:
            print(f"   textexpert-importance analysisfailed: {e}")

    print(f"\n Top-Kanti-polarizationtexttraining completed!")
    print(f" best test AUC: {best_auc:.4f}")
    print(f" LightGCN graph expert: {'enabled' if model_kwargs.get('use_lightgcn', False) else 'not enabled'}")
    print(f" Top-Kanti-polarization: {'enabled' if hasattr(model, 'topk_config') else 'not detected'}")
    print(f" expert-importance analysis: {'enabled' if enable_expert_analysis else 'not enabled'}")


    if hasattr(model, 'topk_config'):
        print(f"\n Top-Kanti-polarizationtrainingsummary:")
        config = model.topk_config
        print(f"  warm-up phase(0-{config['warmup_epochs']}): K={config['warmup_k']}, high exploration")
        print(f"  balancing phase({config['warmup_epochs']+1}-{config['balance_epochs']}): K={config['balance_k']}, dynamictext")
        print(f"  convergence phase({config['balance_epochs']+1}+): K={config['final_k']}, fine tuning")
        print(f"  weight protection: from{config['warmup_max_weight']}progressiveto{config['final_max_weight']}")
        print(f"   expertimportance: every{expert_analysis_interval}epochanalysisonce,textanalysis")


    history_path = save_path.replace('.pt', '_history.csv')
    pd.DataFrame(train_history).to_csv(history_path, index=False)
    print(f" training history saved to: {history_path}")
    print(f" training strategy: Top-Kanti-polarizationtextstreaming(textfiletext+text)")
    print(f" featurestext: full embedding+LightGCN graph expert+Top-Kanti-polarization+expert-importance analysis(image256dim + text3840dim + ID928dim + graph embeddings)")
    print(f" text: textmemory optimization,supportstextdataset")
    print(f" graph expertarchitecture: LightGCNtextscenetexttraininggraph embeddings")
    print(f" anti-polarizationarchitecture: dynamicTop-Ktext+text")
    print(f" expert analysisarchitecture: trainingtextreal-time monitoringexpertimportance")
    print_memory_usage("training completed")

    return best_auc




class MemoryMappedDataset:

    def __init__(self, mmap_data):
        print(" compatibilitytext:textmappingtextforstreamingtext")


        self.length = len(mmap_data.get('labels', []))
        self.temp_dir = mmap_data.get('_temp_dir')


        self.image_embeddings = mmap_data.get('image_embeddings')
        self.text_embeddings = mmap_data.get('text_embeddings')
        self.id_embeddings = mmap_data.get('id_embeddings')
        self.labels = mmap_data.get('labels')
        self.scenes = mmap_data.get('scenes')
        self.user_ids = mmap_data.get('user_ids', [])
        self.item_ids = mmap_data.get('item_ids', [])

        print(f" compatibilitydataset: {self.length:,} text(supportsLightGCN graph expert+Top-Kanti-polarization)")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            if hasattr(self.image_embeddings, '__getitem__'):
                image = torch.from_numpy(self.image_embeddings[idx].copy() if hasattr(self.image_embeddings[idx], 'copy') else self.image_embeddings[idx]).float()
                text = torch.from_numpy(self.text_embeddings[idx].copy() if hasattr(self.text_embeddings[idx], 'copy') else self.text_embeddings[idx]).float()
                id_emb = torch.from_numpy(self.id_embeddings[idx].copy() if hasattr(self.id_embeddings[idx], 'copy') else self.id_embeddings[idx]).float()
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
                scene = torch.tensor(self.scenes[idx], dtype=torch.long)
            else:

                image = torch.tensor(self.image_embeddings[idx], dtype=torch.float32)
                text = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
                id_emb = torch.tensor(self.id_embeddings[idx], dtype=torch.float32)
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
                scene = torch.tensor(self.scenes[idx], dtype=torch.long)


            user_id = str(self.user_ids[idx]) if idx < len(self.user_ids) else "default_user"
            item_id = str(self.item_ids[idx]) if idx < len(self.item_ids) else "default_item"

            return image, text, id_emb, label, scene, user_id, item_id
        except Exception as e:
            print(f" reading sample {idx}  failed: {e}")

            return (
                torch.zeros(256, dtype=torch.float32),
                torch.zeros(3840, dtype=torch.float32),
                torch.zeros(928, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor(0, dtype=torch.long),
                "default_user",
                "default_item"
            )

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                print(f" compatibilitytextfiletext: {self.temp_dir}")
            except Exception as e:
                print(f" textfailed: {e}")

    def __del__(self):
        self.cleanup()