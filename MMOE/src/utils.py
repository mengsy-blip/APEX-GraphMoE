
import os
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import glob
from tqdm import tqdm
import pickle


class H5Dataset_FullEmbeddings(Dataset):
    def __init__(self, h5_files, max_samples_per_file=None):
        self.h5_files = h5_files if isinstance(h5_files, list) else [h5_files]
        self.file_sample_counts = []
        self.cumulative_counts = [0]


        print(" analysisH5file(full-embedding mode+LightGCN graph expert)...")
        for i, h5_file in enumerate(self.h5_files):
            print(f"  processing file [{i+1}/{len(self.h5_files)}]: {os.path.basename(h5_file)}")
            try:
                with h5py.File(h5_file, 'r') as f:
                    num_samples = f['labels'].shape[0]
                    if max_samples_per_file:
                        num_samples = min(num_samples, max_samples_per_file)
                    self.file_sample_counts.append(num_samples)
                    self.cumulative_counts.append(self.cumulative_counts[-1] + num_samples)
                    print(f"    number of samples: {num_samples:,}")
            except Exception as e:
                print(f"     file read error: {e}")
                self.file_sample_counts.append(0)
                self.cumulative_counts.append(self.cumulative_counts[-1])

        self.total_samples = self.cumulative_counts[-1]
        print(f" dataset initializationcompleted(full-embedding mode+LightGCN graph expert): {len(self.h5_files)} file, total {self.total_samples:,} text")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")


        file_idx = 0
        while file_idx < len(self.cumulative_counts) - 1:
            if idx < self.cumulative_counts[file_idx + 1]:
                break
            file_idx += 1

        local_idx = idx - self.cumulative_counts[file_idx]


        try:
            with h5py.File(self.h5_files[file_idx], 'r') as f:
                sample = {}


                if 'image_embeddings' in f:
                    sample['image_embeddings'] = torch.tensor(f['image_embeddings'][local_idx], dtype=torch.float32)
                else:
                    sample['image_embeddings'] = torch.zeros(256, dtype=torch.float32)


                text_fields = ['item_title', 'item_entity_names', 'bill_entity_seq', 'service_entity_seq', 'query_entity_seq']
                text_embeddings = []
                for field in text_fields:
                    field_key = f'{field}_embeddings'
                    if field_key in f:
                        text_embeddings.append(f[field_key][local_idx])
                    else:
                        text_embeddings.append(np.zeros(768, dtype=np.float32))


                sample['text_embeddings'] = torch.tensor(np.concatenate(text_embeddings), dtype=torch.float32)


                id_embeddings = []


                for id_field in ['user_id', 'item_id']:
                    field_key = f'{id_field}_embeddings'
                    if field_key in f:
                        id_embeddings.append(f[field_key][local_idx])
                    else:
                        id_embeddings.append(np.zeros(32, dtype=np.float32))


                for i in range(27):
                    field_key = f'deep_features_{i}_embeddings'
                    if field_key in f:
                        id_embeddings.append(f[field_key][local_idx])
                    else:
                        id_embeddings.append(np.zeros(32, dtype=np.float32))


                sample['id_embeddings'] = torch.tensor(np.concatenate(id_embeddings), dtype=torch.float32)


                sample['label'] = torch.tensor(f['labels'][local_idx], dtype=torch.float32)


                metadata = f['metadata'][local_idx]
                user_id_str = metadata[0].decode('utf-8') if isinstance(metadata[0], bytes) else str(metadata[0])
                item_id_str = metadata[1].decode('utf-8') if isinstance(metadata[1], bytes) else str(metadata[1])
                scene_str = metadata[2].decode('utf-8') if isinstance(metadata[2], bytes) else str(metadata[2])


                sample['user_id'] = user_id_str
                sample['item_id'] = item_id_str


                try:
                    scene_id = int(scene_str) if scene_str.isdigit() else 0
                except:
                    scene_id = 0
                sample['scene'] = torch.tensor(scene_id, dtype=torch.long)

                return sample

        except Exception as e:
            print(f" reading sample {idx}  failed: {e}")

            return self._get_default_sample()

    def _get_default_sample(self):
        return {
            'image_embeddings': torch.zeros(256, dtype=torch.float32),
            'text_embeddings': torch.zeros(3840, dtype=torch.float32),  # 5*768
            'id_embeddings': torch.zeros(928, dtype=torch.float32),     # 29*32
            'label': torch.tensor(0.0, dtype=torch.float32),
            'scene': torch.tensor(0, dtype=torch.long),
            'user_id': "default_user",
            'item_id': "default_item",
        }


def find_h5_files(data_dir, pattern="*_embeddings.h5"):
    if not os.path.exists(data_dir):
        print(f" directorytext: {data_dir}")
        return []

    h5_files = glob.glob(os.path.join(data_dir, pattern))
    h5_files.sort()

    print(f" text {data_dir} textto {len(h5_files)} H5file:")
    for i, file_path in enumerate(h5_files):
        file_size = os.path.getsize(file_path) / (1024**3)  # GB
        print(f"  [{i+1}] {os.path.basename(file_path)} ({file_size:.2f} GB)")

    return h5_files


def analyze_h5_file_full_embeddings(h5_file_path):
    print(f" analysis file(full-embedding mode+LightGCN graph expert): {os.path.basename(h5_file_path)}")

    try:
        with h5py.File(h5_file_path, 'r') as f:
            print(f" filetext:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")

            print(f"\n dataset:")


            image_dim = 0
            text_dims = {}
            id_dims = {}

            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    shape = f[key].shape
                    dtype = f[key].dtype
                    size_mb = f[key].nbytes / (1024**2)

                    print(f"   {key}: {shape} ({dtype}) - {size_mb:.1f} MB")


                    if key == 'image_embeddings':
                        image_dim = shape[1] if len(shape) > 1 else 0
                    elif key.endswith('_embeddings') and any(text_field in key for text_field in ['item_title', 'item_entity_names', 'bill_entity_seq', 'service_entity_seq', 'query_entity_seq']):
                        field_name = key.replace('_embeddings', '')
                        text_dims[field_name] = shape[1] if len(shape) > 1 else 0
                    elif key.endswith('_embeddings') and ('user_id' in key or 'item_id' in key or 'deep_features' in key):
                        field_name = key.replace('_embeddings', '')
                        id_dims[field_name] = shape[1] if len(shape) > 1 else 0

                elif isinstance(f[key], h5py.Group):
                    print(f"   {key}: Group with {len(f[key])} items")

            print(f"\n featuresdimtext:")
            print(f"   image embeddings: {image_dim}dim")

            total_text_dim = sum(text_dims.values())
            print(f"   text embeddings: {total_text_dim}dim (total{len(text_dims)}fields)")
            for field, dim in text_dims.items():
                print(f"    - {field}: {dim}dim")

            total_id_dim = sum(id_dims.values())
            print(f"   ID embeddings: {total_id_dim}dim (total{len(id_dims)}features)")
            for field, dim in sorted(id_dims.items()):
                print(f"    - {field}: {dim}dim")

            print(f"   total feature dimension: {image_dim + total_text_dim + total_id_dim}dim")
            print(f"   + LightGCNgraph embeddings: dynamic dimension(textscenetext)")


            if 'metadata' in f and f['metadata'].shape[0] > 0:
                print(f"\n textexample (first3):")
                for i in range(min(3, f['metadata'].shape[0])):
                    metadata = f['metadata'][i]
                    user_id = metadata[0].decode('utf-8') if isinstance(metadata[0], bytes) else str(metadata[0])
                    item_id = metadata[1].decode('utf-8') if isinstance(metadata[1], bytes) else str(metadata[1])
                    scene = metadata[2].decode('utf-8') if isinstance(metadata[2], bytes) else str(metadata[2])
                    label = f['labels'][i] if 'labels' in f else 'N/A'
                    print(f"  text {i}: user_id={user_id}, item_id={item_id}, scene={scene}, label={label}")
                    print(f"     LightGCNtextusinguser_idanditem_idtextgraph embeddings")

    except Exception as e:
        print(f" analysis file failed: {e}")


def check_lightgcn_embeddings(gcn_logs_dir="gcn_logs_incremental"):
    print(f" checkingLightGCNgraph embeddingsfile...")
    print(f"  directory: {gcn_logs_dir}")

    result = {
        'available': False,
        'scenes': {},
        'total_size_gb': 0,
        'error_msg': None
    }

    if not os.path.exists(gcn_logs_dir):
        result['error_msg'] = f"LightGCN directory does not exist: {gcn_logs_dir}"
        print(f" {result['error_msg']}")
        return result

    try:

        scene_dirs = [d for d in os.listdir(gcn_logs_dir) if d.startswith('scene_')]
        scene_dirs.sort()

        print(f"  scene directories found: {len(scene_dirs)} ")

        for scene_dir in scene_dirs:
            scene_path = os.path.join(gcn_logs_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            try:
                scene_id = int(scene_dir.split('_')[1])
            except (IndexError, ValueError):
                print(f"     invalid scene directory format: {scene_dir}")
                continue


            user_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_user_emb_final.npy")
            item_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_item_emb_final.npy")

            scene_info = {
                'user_emb_available': os.path.exists(user_emb_path),
                'item_emb_available': os.path.exists(item_emb_path),
                'user_emb_size_gb': 0,
                'item_emb_size_gb': 0,
                'user_shape': None,
                'item_shape': None,
                'emb_dim': 0
            }

            if scene_info['user_emb_available']:
                scene_info['user_emb_size_gb'] = os.path.getsize(user_emb_path) / (1024**3)
                try:
                    user_emb = np.load(user_emb_path, mmap_mode='r')
                    scene_info['user_shape'] = user_emb.shape
                    scene_info['emb_dim'] = user_emb.shape[1] if len(user_emb.shape) > 1 else 0
                except Exception as e:
                    print(f"     failed to read user embeddings: {e}")

            if scene_info['item_emb_available']:
                scene_info['item_emb_size_gb'] = os.path.getsize(item_emb_path) / (1024**3)
                try:
                    item_emb = np.load(item_emb_path, mmap_mode='r')
                    scene_info['item_shape'] = item_emb.shape
                    if scene_info['emb_dim'] == 0:
                        scene_info['emb_dim'] = item_emb.shape[1] if len(item_emb.shape) > 1 else 0
                except Exception as e:
                    print(f"     failed to read item embeddings: {e}")


            mapping_files = ['user_id_to_idx.pkl', 'item_id_to_idx.pkl', 'idx_to_user_id.pkl', 'idx_to_item_id.pkl']
            scene_info['mapping_files'] = {}
            for mapping_file in mapping_files:
                mapping_path = os.path.join(scene_path, mapping_file)
                scene_info['mapping_files'][mapping_file] = os.path.exists(mapping_path)

            result['scenes'][scene_id] = scene_info
            result['total_size_gb'] += scene_info['user_emb_size_gb'] + scene_info['item_emb_size_gb']


            status = "" if (scene_info['user_emb_available'] and scene_info['item_emb_available']) else ""
            print(f"    {status} scene{scene_id}:")
            if scene_info['user_emb_available']:
                print(f"      user embeddings: {scene_info['user_shape']} ({scene_info['user_emb_size_gb']:.3f} GB)")
            else:
                print(f"      user embeddings: missing")

            if scene_info['item_emb_available']:
                print(f"      item embeddings: {scene_info['item_shape']} ({scene_info['item_emb_size_gb']:.3f} GB)")
            else:
                print(f"      item embeddings: missing")

            if scene_info['emb_dim'] > 0:
                print(f"      embedding dimension: {scene_info['emb_dim']}")


            available_mappings = sum(scene_info['mapping_files'].values())
            print(f"      ID mapping files: {available_mappings}/{len(mapping_files)} text")


        available_scenes = [s for s, info in result['scenes'].items()
                          if info['user_emb_available'] and info['item_emb_available']]

        if available_scenes:
            result['available'] = True
            print(f"\n LightGCNgraph embeddingscheckingcompleted:")
            print(f"  available scenes: {len(available_scenes)}  {available_scenes}")
            print(f"  text: {result['total_size_gb']:.3f} GB")
        else:
            result['error_msg'] = "texttotextscenegraph embeddings"
            print(f"\n {result['error_msg']}")

    except Exception as e:
        result['error_msg'] = f"checkingLightGCNgraph embeddings failed: {e}"
        print(f" {result['error_msg']}")

    return result


def split_h5_files(h5_files, train_ratio=0.8):

    train_files = []
    test_files = []

    for h5_file in h5_files:
        file_name = os.path.basename(h5_file).lower()
        if 'train' in file_name:
            train_files.append(h5_file)
        elif 'test' in file_name:
            test_files.append(h5_file)
        else:

            if len(train_files) < len(h5_files) * train_ratio:
                train_files.append(h5_file)
            else:
                test_files.append(h5_file)

    print(f" datasettext:")
    print(f"  training files: {len(train_files)} ")
    for f in train_files:
        print(f"    - {os.path.basename(f)}")
    print(f"  test files: {len(test_files)} ")
    for f in test_files:
        print(f"    - {os.path.basename(f)}")

    return train_files, test_files


def get_h5_stats_full_embeddings(h5_files):
    stats = {
        'total_files': len(h5_files),
        'total_samples': 0,
        'positive_samples': 0,
        'negative_samples': 0,
        'total_size_gb': 0,
        'feature_dimensions': {
            'image': 256,
            'text': 3840,  # 5*768
            'id': 928,     # 29*32
            'graph': 'dynamic'
        },
        'total_feature_dim': 256 + 3840 + 928,
        'lightgcn_info': None
    }

    print(" textH5filetext(full-embedding mode+LightGCN graph expert)...")

    for h5_file in tqdm(h5_files, desc="processing file"):
        try:
            file_size_gb = os.path.getsize(h5_file) / (1024**3)
            stats['total_size_gb'] += file_size_gb

            with h5py.File(h5_file, 'r') as f:

                if 'num_samples' in f.attrs:
                    stats['total_samples'] += f.attrs['num_samples']
                else:

                    if 'labels' in f:
                        stats['total_samples'] += f['labels'].shape[0]

                if 'positive_samples' in f.attrs:
                    stats['positive_samples'] += f.attrs['positive_samples']

                if 'negative_samples' in f.attrs:
                    stats['negative_samples'] += f.attrs['negative_samples']

        except Exception as e:
            print(f"   processing file {h5_file}  failed: {e}")
            continue


    total_labeled = stats['positive_samples'] + stats['negative_samples']
    if total_labeled > 0:
        stats['positive_ratio'] = stats['positive_samples'] / total_labeled
        stats['negative_ratio'] = stats['negative_samples'] / total_labeled
    else:
        stats['positive_ratio'] = 0.0
        stats['negative_ratio'] = 0.0


    lightgcn_info = check_lightgcn_embeddings()
    stats['lightgcn_info'] = lightgcn_info
    if lightgcn_info['available']:
        stats['total_size_gb'] += lightgcn_info['total_size_gb']


        first_scene = next(iter(lightgcn_info['scenes'].values()), {})
        if first_scene.get('emb_dim', 0) > 0:
            stats['feature_dimensions']['graph'] = first_scene['emb_dim']

    return stats


def print_training_recommendations_full_embeddings(stats):
    print("\n textGPUtrainingsuggestion(full-embedding mode+LightGCN graph expert):")


    if stats['total_samples'] > 5000000:
        print("   text (>5M text):")
        print("    - batch_size: 512-1024 (featuresdimtext)")
        print("    - learning_rate: 3e-5 - 5e-5")
        print("    - usingmixed precisiontraining")
        print("    - text")
        print("    -  LightGCN graph experttext")
    elif stats['total_samples'] > 1000000:
        print("   text (1M-5M text):")
        print("    - batch_size: 1024-2048")
        print("    - learning_rate: 5e-5 - 1e-4")
        print("    - texttraining configuration")
        print("    -  LightGCN graph experttextimprovementtext")
    else:
        print("   text (<1M text):")
        print("    - batch_size: 256-512")
        print("    - learning_rate: 1e-4 - 2e-4")
        print("    - textepochs")
        print("    -  LightGCN graph experttext")


    total_dim = stats['total_feature_dim']
    print(f"\n featuresdimtextanalysis:")
    print(f"    - conventional feature dimension: {total_dim}dim")
    print(f"    - imagefeatures: {stats['feature_dimensions']['image']}dim")
    print(f"    - textfeatures: {stats['feature_dimensions']['text']}dim")
    print(f"    - IDfeatures: {stats['feature_dimensions']['id']}dim")


    lightgcn_info = stats.get('lightgcn_info')
    if lightgcn_info and lightgcn_info['available']:
        graph_dim = stats['feature_dimensions'].get('graph', 'dynamic')
        print(f"    -  LightGCNtextfeatures: {graph_dim}dim")
        print(f"    -  available scenes: {len(lightgcn_info['scenes'])} ")
        print(f"    -  graph embeddingstext: {lightgcn_info['total_size_gb']:.3f} GB")
        print(f"    - featurestext: text (multimodal+texttext+textID+text)")
    else:
        print(f"    -  LightGCN graph expert: text")
        print(f"    - featurestext: text (multimodal+texttext+textID)")
        if lightgcn_info and lightgcn_info.get('error_msg'):
            print(f"    - error message: {lightgcn_info['error_msg']}")


    base_memory_estimate = (stats['total_samples'] * total_dim * 4) / (1024**3)  # 4 bytes per float32
    lightgcn_memory = lightgcn_info['total_size_gb'] if (lightgcn_info and lightgcn_info['available']) else 0

    print(f"\n memory estimate:")
    print(f"    - textfeaturestext: ~{base_memory_estimate:.2f} GB")
    if lightgcn_memory > 0:
        print(f"    -  LightGCNgraph embeddings: ~{lightgcn_memory:.3f} GB")
        print(f"    - textmemory requirement: ~{base_memory_estimate + lightgcn_memory:.2f} GB")
    else:
        print(f"    - textmemory requirement: ~{base_memory_estimate:.2f} GB")
    print(f"    - suggestionusingFP16textmemory usage")
    print(f"    - textmappingtext")


    if stats['total_size_gb'] > 20:
        print("   textfile (>20GB):")
        print("    - textnum_workers (4-8)")
        print("    - usingpin_memory=True")
        print("    - text")
        print("    -  enabledtextmappingtext")


    if 'positive_ratio' in stats and stats['positive_ratio'] > 0:
        if stats['positive_ratio'] < 0.1 or stats['positive_ratio'] > 0.9:
            print("   text:")
            print("    - textusingtext")
            print("    - textpositive/negativetext")
            print(f"    - textfirstpositive samplestext: {stats['positive_ratio']:.2%}")
            print("    -  LightGCN graph experttext")


    print(f"\n textarchitecturesuggestion:")
    print(f"    - usingtextexperttext")
    print(f"    - image expert: CNN/Transformer")
    print(f"    - text expert: text")
    print(f"    - IDexpert: MLP/Embedding")

    if lightgcn_info and lightgcn_info['available']:
        print(f"    -  LightGCN graph expert: textuser-itemtext")
        print(f"    -  graph experttext: text")
        print(f"    -  textstrategy: textexpert")
        print(f"    -  textimprovement: AUC +2~5%")
    else:
        print(f"    -  LightGCN graph experttext,suggestion:")
        print(f"      - checkinggcn_logs_incrementaldirectory")
        print(f"      - textLightGCNtexttrainingtext")
        print(f"      - ordisabledgraph expert (use_lightgcn=False)")

    print(f"    - text: textortext")


def quick_h5_check_full_embeddings(data_dir_or_files):
    if isinstance(data_dir_or_files, str):
        h5_files = find_h5_files(data_dir_or_files)
    else:
        h5_files = data_dir_or_files

    if not h5_files:
        print(" texttoH5file")
        return False


    stats = get_h5_stats_full_embeddings(h5_files[:3])

    print(f"\n data overview(full-embedding mode+LightGCN graph expert):")
    print(f"  number of files: {len(h5_files)}")
    print(f"  number of samplestext: {stats['total_samples']:,}")
    print(f"  data size: {stats['total_size_gb']:.2f} GB")
    print(f"  conventional feature dimension: {stats['total_feature_dim']}dim")


    lightgcn_info = stats.get('lightgcn_info')
    if lightgcn_info:
        if lightgcn_info['available']:
            print(f"   LightGCN graph expert:  text")
            print(f"    - available scenes: {len(lightgcn_info['scenes'])} ")
            print(f"    - graph embeddingstext: {lightgcn_info['total_size_gb']:.3f} GB")
            if lightgcn_info['scenes']:
                first_scene = next(iter(lightgcn_info['scenes'].values()))
                if first_scene.get('emb_dim', 0) > 0:
                    print(f"    - textembedding dimension: {first_scene['emb_dim']}dim")
        else:
            print(f"   LightGCN graph expert:  text")
            if lightgcn_info.get('error_msg'):
                print(f"    - reason: {lightgcn_info['error_msg']}")

    if stats['positive_samples'] > 0:
        print(f"  positive samples: {stats['positive_samples']:,} ({stats['positive_ratio']:.2%})")
        print(f"  negative samples: {stats['negative_samples']:,} ({stats['negative_ratio']:.2%})")


    print_training_recommendations_full_embeddings(stats)

    return True


def create_h5_dataset_full_embeddings(h5_files, **kwargs):
    return H5Dataset_FullEmbeddings(
        h5_files=h5_files,
        **kwargs
    )


def estimate_feature_importance():
    weights = {
        'image_weight': 0.25,
        'text_weight': 0.4,
        'id_weight': 0.15,
        'graph_weight': 0.2,
        'fusion_strategy': 'mmoe_gating'
    }

    return weights


def compare_feature_usage(num_samples, feature_dims):
    scenarios = {
        'image only': feature_dims['image'],
        'text only': feature_dims['text'],
        'ID only': feature_dims['id'],
        'image+text': feature_dims['image'] + feature_dims['text'],
        'all conventional features': feature_dims['image'] + feature_dims['text'] + feature_dims['id'],
        ' all features+LightGCN': feature_dims['image'] + feature_dims['text'] + feature_dims['id']
    }

    print(f"\n featurestext ({num_samples:,} text):")
    for scenario, dim in scenarios.items():
        if 'LightGCN' in scenario:
            memory_gb = (num_samples * dim * 4) / (1024**3)
            print(f"  {scenario}: {dim}dim + dynamicgraph embeddings -> {memory_gb:.2f} GB + graph embeddings")
        else:
            memory_gb = (num_samples * dim * 4) / (1024**3)  # 4 bytes per float32
            print(f"  {scenario}: {dim}dim -> {memory_gb:.2f} GB")

    print(f"\n suggestion:")
    print(f"  - fromtextfeaturestextstartingtext")
    print(f"  - textfeaturestext")
    print(f"  - usingfeaturestextanalysistext")
    print(f"  -  LightGCN graph experttextimprovement")
    print(f"  -  text,textdisabledgraph expert")


def validate_lightgcn_compatibility(h5_files, gcn_logs_dir="gcn_logs_incremental"):
    print(f" textH5textandLightGCNgraph embeddingscompatibility...")

    result = {
        'compatible': False,
        'h5_scenes': set(),
        'lightgcn_scenes': set(),
        'missing_scenes': set(),
        'sample_user_ids': [],
        'sample_item_ids': [],
        'compatibility_issues': []
    }


    print("  checkingH5filetextsceneanduser/itemID...")
    for h5_file in h5_files[:2]:
        if not os.path.exists(h5_file):
            continue

        try:
            with h5py.File(h5_file, 'r') as f:
                if 'metadata' not in f:
                    result['compatibility_issues'].append(f"filemissingmetadata: {h5_file}")
                    continue


                sample_size = min(100, f['metadata'].shape[0])
                metadata = f['metadata'][:sample_size]

                for i in range(sample_size):
                    meta = metadata[i]
                    user_id = meta[0].decode('utf-8') if isinstance(meta[0], bytes) else str(meta[0])
                    item_id = meta[1].decode('utf-8') if isinstance(meta[1], bytes) else str(meta[1])
                    scene_str = meta[2].decode('utf-8') if isinstance(meta[2], bytes) else str(meta[2])

                    try:
                        scene_id = int(scene_str) if scene_str.isdigit() else 0
                        result['h5_scenes'].add(scene_id)
                    except:
                        result['compatibility_issues'].append(f"invalid scene ID: {scene_str}")

                    if len(result['sample_user_ids']) < 10:
                        result['sample_user_ids'].append(user_id)
                    if len(result['sample_item_ids']) < 10:
                        result['sample_item_ids'].append(item_id)

        except Exception as e:
            result['compatibility_issues'].append(f"readingH5filefailed {h5_file}: {e}")


    lightgcn_info = check_lightgcn_embeddings(gcn_logs_dir)
    if lightgcn_info['available']:
        result['lightgcn_scenes'] = set(lightgcn_info['scenes'].keys())


        result['missing_scenes'] = result['h5_scenes'] - result['lightgcn_scenes']

        if result['missing_scenes']:
            result['compatibility_issues'].append(f"missingscenegraph embeddings: {result['missing_scenes']}")


        for scene_id in result['lightgcn_scenes']:
            if scene_id not in lightgcn_info['scenes']:
                continue

            scene_path = os.path.join(gcn_logs_dir, f"scene_{scene_id}")
            user_mapping_path = os.path.join(scene_path, "user_id_to_idx.pkl")
            item_mapping_path = os.path.join(scene_path, "item_id_to_idx.pkl")

            if os.path.exists(user_mapping_path) and os.path.exists(item_mapping_path):
                try:
                    with open(user_mapping_path, 'rb') as f:
                        user_mapping = pickle.load(f)
                    with open(item_mapping_path, 'rb') as f:
                        item_mapping = pickle.load(f)


                    missing_users = [uid for uid in result['sample_user_ids'][:5] if uid not in user_mapping]
                    missing_items = [iid for iid in result['sample_item_ids'][:5] if iid not in item_mapping]

                    if missing_users:
                        result['compatibility_issues'].append(f"scene{scene_id}missinguserIDmapping: {missing_users[:3]}...")
                    if missing_items:
                        result['compatibility_issues'].append(f"scene{scene_id}missingitemIDmapping: {missing_items[:3]}...")

                except Exception as e:
                    result['compatibility_issues'].append(f"scene{scene_id}readingIDmappingfailed: {e}")
            else:
                result['compatibility_issues'].append(f"scene{scene_id}missingID mapping files")


        common_scenes = result['h5_scenes'] & result['lightgcn_scenes']
        if common_scenes and len(result['compatibility_issues']) == 0:
            result['compatible'] = True
        elif common_scenes:
            result['compatible'] = True
    else:
        result['compatibility_issues'].append("LightGCNgraph embeddings unavailable")


    print(f"\n compatibility checktext:")
    print(f"  H5filescene: {sorted(result['h5_scenes'])}")
    print(f"  LightGCNscene: {sorted(result['lightgcn_scenes'])}")
    if result['missing_scenes']:
        print(f"   missing scenes: {sorted(result['missing_scenes'])}")

    if result['compatible']:
        print(f"   compatibility: text")
        if result['compatibility_issues']:
            print(f"   warning: {len(result['compatibility_issues'])} text")
            for issue in result['compatibility_issues'][:3]:
                print(f"    - {issue}")
    else:
        print(f"   compatibility: text")
        for issue in result['compatibility_issues']:
            print(f"    - {issue}")

    return result



def batch_analyze_h5_files_full_embeddings(h5_files, max_files=5):
    print(f" textanalysisH5file(full-embedding mode+LightGCN graph expert,text{max_files})...")

    files_to_analyze = h5_files[:max_files]

    for i, h5_file in enumerate(files_to_analyze):
        print(f"\n{'='*50}")
        print(f"analysis file {i+1}/{len(files_to_analyze)}")
        analyze_h5_file_full_embeddings(h5_file)

    if len(h5_files) > max_files:
        print(f"\n text {len(h5_files) - max_files} filetextanalysis")

    print(f"\n text:")
    stats = get_h5_stats_full_embeddings(h5_files)
    print(f"  textfiletext: {stats['total_files']}")
    print(f"  textnumber of samples: {stats['total_samples']:,}")
    print(f"  text: {stats['total_size_gb']:.2f} GB")
    print(f"  conventional feature dimension: {stats['total_feature_dim']}dim")


    lightgcn_info = stats.get('lightgcn_info')
    if lightgcn_info and lightgcn_info['available']:
        print(f"   LightGCN graph expert: text ({len(lightgcn_info['scenes'])}scene)")
    else:
        print(f"   LightGCN graph expert: text")


    compare_feature_usage(stats['total_samples'], stats['feature_dimensions'])


    if h5_files:
        print(f"\n LightGCNcompatibility check:")
        compatibility = validate_lightgcn_compatibility(h5_files)
        if compatibility['compatible']:
            print("   H5textandLightGCNgraph embeddingstext")
        else:
            print("   textcompatibility issues,suggestioncheckingtext")



def prepare_lightgcn_for_training(gcn_logs_dir="gcn_logs_incremental", target_scenes=[0, 1, 2, 3, 4]):
    print(f" fortrainingpreparingLightGCNgraph embeddings...")
    print(f"  directory: {gcn_logs_dir}")
    print(f"  target scenes: {target_scenes}")

    if not os.path.exists(gcn_logs_dir):
        print(f" LightGCN directory does not exist: {gcn_logs_dir}")
        print(" textLightGCNtexttrainingtextgraph embeddings")
        return False

    lightgcn_info = check_lightgcn_embeddings(gcn_logs_dir)

    if not lightgcn_info['available']:
        print(f" LightGCNgraph embeddings unavailable")
        if lightgcn_info.get('error_msg'):
            print(f"   reason: {lightgcn_info['error_msg']}")
        return False

    available_scenes = set(lightgcn_info['scenes'].keys())
    missing_scenes = set(target_scenes) - available_scenes

    if missing_scenes:
        print(f" missingscenegraph embeddings: {sorted(missing_scenes)}")
        print(f"   available scenes: {sorted(available_scenes)}")
        print(" suggestion:")
        print("   1. formissing scenestextLightGCNtexttraining")
        print("   2. ortexttarget_scenesparameters")
        return False

    print(f" LightGCNgraph embeddingspreparingcompleted")
    print(f"   available scenes: {sorted(available_scenes)}")
    print(f"   graph embeddingstext: {lightgcn_info['total_size_gb']:.3f} GB")

    return True