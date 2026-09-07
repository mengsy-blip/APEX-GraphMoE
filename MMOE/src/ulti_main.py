
import os
import torch
import torch.nn as nn
import numpy as np
import random
import warnings
import psutil
import gc
warnings.filterwarnings('ignore')


from multi_mmoe import HeterogeneousMMoE_FullEmbeddings, create_heterogeneous_mmoe_model
from multi_model_train import (
    train_model_memory_single_gpu_full_embeddings,
    load_single_h5_file_to_memory_full_embeddings,
    print_memory_usage
)
from utils import find_h5_files


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_memory_requirements_heterogeneous(train_files, test_files, max_samples=None):
    print(f" checkingheterogeneous experttextmemory requirement...")


    memory_info = psutil.virtual_memory()
    total_memory_gb = memory_info.total / (1024**3)
    available_memory_gb = memory_info.available / (1024**3)

    print(f" system memory:")
    print(f"  total memory: {total_memory_gb:.1f} GB")
    print(f"  text: {available_memory_gb:.1f} GB")


    max_file_memory = 0
    total_samples = 0

    sample_files = (train_files + test_files)[:3]

    for i, h5_file in enumerate(sample_files):
        if not os.path.exists(h5_file):
            continue

        try:
            import h5py
            with h5py.File(h5_file, 'r') as f:
                file_samples = f['labels'].shape[0]
                if max_samples and file_samples > max_samples:
                    file_samples = max_samples


                sample_size_kb = 25
                file_memory_gb = (file_samples * sample_size_kb) / (1024 * 1024)

                max_file_memory = max(max_file_memory, file_memory_gb)
                total_samples += file_samples

                print(f"  file {i+1}: {file_samples:,} text -> ~{file_memory_gb:.2f} GB")

        except Exception as e:
            print(f"   file {i+1} checkingfailed: {e}")


    lightgcn_memory_gb = 0
    gcn_logs_dir = "/root/megrez-tmp/code/baseline+lightgcn/traininglightgcn/gcn_logs_incremental"
    if os.path.exists(gcn_logs_dir):
        print(f" checkingLightGCNgraph embeddingsmemory requirement...")
        scene_dirs = [d for d in os.listdir(gcn_logs_dir) if d.startswith('scene_')]
        for scene_dir in scene_dirs:
            scene_path = os.path.join(gcn_logs_dir, scene_dir)
            try:
                scene_id = int(scene_dir.split('_')[1])
                user_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_user_emb_final.npy")
                item_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_item_emb_final.npy")

                if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
                    user_size = os.path.getsize(user_emb_path) / (1024**3)
                    item_size = os.path.getsize(item_emb_path) / (1024**3)
                    lightgcn_memory_gb += user_size + item_size
                    print(f"  scene{scene_id}: user embeddings{user_size:.3f}GB + item embeddings{item_size:.3f}GB")
            except:
                continue
        print(f"  LightGCNgraph embeddingstext: {lightgcn_memory_gb:.3f} GB")
    else:
        print(f" LightGCN directory does not exist: {gcn_logs_dir}")


    pytorch_memory_gb = max_file_memory * 1.5
    model_memory_gb = 0.5
    training_overhead_gb = 2.0 + lightgcn_memory_gb

    total_required_gb = pytorch_memory_gb + model_memory_gb + training_overhead_gb

    print(f"\n heterogeneous experttextmemory requirementanalysis:")
    print(f"  textfiletext: ~{max_file_memory:.2f} GB")
    print(f"  PyTorchtext: ~{pytorch_memory_gb:.2f} GB")
    print(f"  heterogeneous experttext: ~{model_memory_gb:.2f} GB")
    print(f"  LightGCNgraph embeddings: ~{lightgcn_memory_gb:.3f} GB")
    print(f"  trainingtext: ~{training_overhead_gb:.2f} GB")
    print(f"  textmemory requirement: ~{total_required_gb:.2f} GB")
    print(f"   heterogeneoustext: experttext,textfeaturestext")


    memory_ratio = total_required_gb / available_memory_gb

    if memory_ratio > 0.9:
        print(f" text! text {total_required_gb:.1f} GB,text {available_memory_gb:.1f} GB")
        return False, total_required_gb
    elif memory_ratio > 0.7:
        print(f" text ({memory_ratio:.1%})")
        return True, total_required_gb
    else:
        print(f" text ({memory_ratio:.1%})")
        return True, total_required_gb


def optimize_memory_settings():
    print(" text...")


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        print(" CUDAmemory optimizationenabled")


    gc.set_threshold(700, 10, 10)
    print(" textenabled")


def check_environment():
    print(" checkingheterogeneous experttrainingtext...")


    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f" CUDAtext: {gpu_name} ({gpu_memory:.1f} GB)")

        if gpu_memory < 8:
            print(" GPUtext,heterogeneous experttrainingsuggestion:")
            print("  - reducebatch_sizeto256-512")
            print("  - enabledmixed precision")
            print("  - disabledtextexpert")
        else:
            print(" GPUtext,textheterogeneous experttraining")
    else:
        print(" CUDAtext,textusingCPU(text)")


    gcn_logs_dir = "/root/megrez-tmp/code/baseline+lightgcn/traininglightgcn/gcn_logs_incremental"
    if os.path.exists(gcn_logs_dir):
        scene_dirs = [d for d in os.listdir(gcn_logs_dir) if d.startswith('scene_')]
        print(f" LightGCN graph expertchecking:")
        print(f"  directory: {gcn_logs_dir}")
        print(f"  textscene: {len(scene_dirs)} ")

        available_scenes = 0
        for scene_dir in sorted(scene_dirs):
            try:
                scene_id = int(scene_dir.split('_')[1])
                scene_path = os.path.join(gcn_logs_dir, scene_dir)
                user_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_user_emb_final.npy")
                item_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_item_emb_final.npy")

                if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
                    print(f"     scene{scene_id}: graph embeddingstext")
                    available_scenes += 1
                else:
                    print(f"     scene{scene_id}: graph embeddingsmissing")
            except:
                print(f"     invalid scene directory format: {scene_dir}")

        if available_scenes > 0:
            print(" LightGCN graph experttext")
        else:
            print(" LightGCN graph experttext")
    else:
        print(f" LightGCN directory does not exist: {gcn_logs_dir}")

    return True


def prepare_data_files(train_data_dir, test_data_dir, train_files=None, test_files=None):
    print(" preparingtextfile...")


    if train_files:
        train_files = [f for f in train_files if os.path.exists(f)]
        print(f" usingtexttraining files: {len(train_files)} ")
    else:
        train_files = find_h5_files(train_data_dir)
        print(f" fromdirectorytexttraining files: {len(train_files)} ")

    if test_files:
        test_files = [f for f in test_files if os.path.exists(f)]
        print(f" usingtexttest files: {len(test_files)} ")
    else:
        test_files = find_h5_files(test_data_dir)
        print(f" fromdirectorytexttest files: {len(test_files)} ")


    if train_data_dir == test_data_dir and not (train_files and test_files):
        all_files = find_h5_files(train_data_dir)
        train_files = [f for f in all_files if 'train' in os.path.basename(f).lower()]
        test_files = [f for f in all_files if 'test' in os.path.basename(f).lower()]

        if not train_files and not test_files:
            print(" textfromfiletexttraining/test files,text")
            all_files.sort()
            mid_point = len(all_files) // 2
            train_files = all_files[:mid_point] if len(all_files) > 1 else all_files
            test_files = all_files[mid_point:] if len(all_files) > 1 else all_files

    if not train_files:
        print(f" texttotrainingtextfile")
        return False, [], []

    if not test_files:
        print(f" texttotestingtextfile")
        return False, [], []

    print(f"\n textfiletext:")
    print(f"  training files: {len(train_files)}  (heterogeneous expertstreaming)")
    total_train_size = 0
    for i, f in enumerate(train_files):
        size_gb = os.path.getsize(f) / (1024**3)
        total_train_size += size_gb
        print(f"    [{i+1}] {os.path.basename(f)} ({size_gb:.2f} GB)")

    print(f"  test files: {len(test_files)}  (heterogeneous expertstreaming)")
    total_test_size = 0
    for i, f in enumerate(test_files):
        size_gb = os.path.getsize(f) / (1024**3)
        total_test_size += size_gb
        print(f"    [{i+1}] {os.path.basename(f)} ({size_gb:.2f} GB)")

    print(f"  total file size: {total_train_size + total_test_size:.2f} GB")
    print(f"   heterogeneous experttext: text,textfeaturestext")

    return True, train_files, test_files


def create_heterogeneous_model_config(base_config):
    print(" textheterogeneous expertmultimodalMMoEtext...")


    model_config = {
        'image_emb_dim': 256,
        'text_emb_dim': 3840,
        'id_emb_dim': 896,

        'use_lightgcn': True,
        'gcn_logs_dir': "/root/megrez-tmp/code/baseline+lightgcn/traininglightgcn/gcn_logs_incremental",
        **base_config
    }

    print(f" heterogeneous experttext:")
    print(f"  image embeddings: {model_config['image_emb_dim']}dim -> image expert")
    print(f"  text embeddings: {model_config['text_emb_dim']}dim -> text expert")
    print(f"  ID embeddings: {model_config['id_emb_dim']}dim -> IDexpert")
    print(f"  textdimtext: {model_config['image_emb_dim'] + model_config['text_emb_dim'] + model_config['id_emb_dim']}dim -> textMLPexpert")
    print(f"   LightGCN graph expert: {'enabled' if model_config['use_lightgcn'] else 'disabled'}")
    if model_config['use_lightgcn']:
        print(f"   LightGCNdirectory: {model_config['gcn_logs_dir']}")

    return HeterogeneousMMoE_FullEmbeddings, model_config


def analyze_expert_performance(model, test_files, device, max_samples=5000):
    print("\n analysisheterogeneous experttext...")

    if not test_files:
        print(" texttest files,textexpert analysis")
        return

    model.eval()

    try:

        test_data = load_single_h5_file_to_memory_full_embeddings(
            test_files[0], max_samples=max_samples
        )


        image_emb = torch.tensor(test_data['image_embeddings'][:100], dtype=torch.float32).to(device)
        text_emb = torch.tensor(test_data['text_embeddings'][:100], dtype=torch.float32).to(device)
        id_emb = torch.tensor(test_data['id_embeddings'][:100], dtype=torch.float32).to(device)
        scenes = torch.tensor(test_data['scenes'][:100], dtype=torch.long).to(device)
        user_ids = test_data['user_ids'][:100]
        item_ids = test_data['item_ids'][:100]

        with torch.no_grad():

            expert_weights = model.get_expert_weights(
                image_emb, text_emb, id_emb, scenes, user_ids, item_ids
            )

            print(" textsceneexpert weightsanalysis:")
            for scene, weights in expert_weights.items():
                print(f"  {scene}:")
                sorted_experts = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                for expert, weight in sorted_experts:
                    print(f"    {expert}: {weight:.3f}")


            contributions = model.get_expert_contributions(
                image_emb, text_emb, id_emb, scenes, user_ids, item_ids
            )

            print("\n expert contributiontextanalysis:")
            for contrib_type, value in contributions.items():
                if not contrib_type.endswith('text'):
                    print(f"  {contrib_type}: {value:.4f}")


            if 'cold startusertext' in contributions:
                cold_ratio = contributions['cold startusertext']
                print(f"\n cold startanalysis:")
                print(f"  cold startusertext: {cold_ratio:.2%}")
                if 'LightGCN graph expert' in contributions and 'graph expert_cold starttext' in contributions:
                    normal_impact = contributions['LightGCN graph expert']
                    cold_impact = contributions['graph expert_cold starttext']
                    print(f"  graph experttext: {normal_impact:.4f}")
                    print(f"  graph expertcold starttext: {cold_impact:.4f}")
                    print(f"  text: {(cold_impact/normal_impact)*100:.1f}%" if normal_impact > 0 else "  text")

    except Exception as e:
        print(f" expert analysisfailed: {e}")


def main():
    print(" startingheterogeneous experttextGPUmultimodalMMoEtraining...")
    print("=" * 90)


    set_seed(42)
    print(" text")


    optimize_memory_settings()


    if not check_environment():
        print(" textcheckingfailed")
        return


    config = {

        'train_data_dir': "/root/megrez-tmp/embeddings_pretrained",
        'test_data_dir': "/root/megrez-tmp/embeddings_pretrained",

       'train_files': [
            "/root/megrez-tmp/embeddings_pretrained/antm2c_10m_part0_train_embeddings.h5",
            "/root/megrez-tmp/embeddings_pretrained/antm2c_10m_part1_train_embeddings.h5",
            "/root/megrez-tmp/embeddings_pretrained/antm2c_10m_part2_train_embeddings.h5",
        ],
        'test_files': [
            "/root/megrez-tmp/embeddings_pretrained/antm2c_10m_part0_test_embeddings.h5",
            "/root/megrez-tmp/embeddings_pretrained/antm2c_10m_part1_test_embeddings.h5",
            "/root/megrez-tmp/embeddings_pretrained/antm2c_10m_part2_test_embeddings.h5",
        ],

        'model_save_dir': "heterogeneous_model_full_embeddings",
        'model_save_name': "heterogeneous_mmoe_full_embeddings_best.pt",


        'epochs': 50,
        'learning_rate': 3e-5,
        'batch_size': 4096,
        'early_stop': 8,
        'max_samples': None,

        'use_mmap': True,


        'model_config': {
            'expert_hidden_dim': 512,
            'tower_hidden_dims': [512, 256, 128],
            'output_dim': 1,
            'dropout': 0.3,
            'num_scenes': 5,

            'use_lightgcn': True,
            'gcn_logs_dir': "/root/megrez-tmp/code/baseline+lightgcn/traininglightgcn/gcn_logs_incremental"
        }
    }

    print(" heterogeneous experttraining configuration:")
    for key, value in config.items():
        if key != 'model_config':
            print(f"  {key}: {value}")

    print(" heterogeneous expertarchitecturetext:")
    for key, value in config['model_config'].items():
        marker = "" if key in ['use_lightgcn', 'gcn_logs_dir'] else "  "
        print(f"  {marker} {key}: {value}")


    success, train_files, test_files = prepare_data_files(
        config['train_data_dir'],
        config['test_data_dir'],
        config.get('train_files'),
        config.get('test_files')
    )

    if not success:
        print(" textfilepreparingfailed")
        return


    print("\n" + "="*60)
    memory_ok, required_memory = check_memory_requirements_heterogeneous(
        train_files, test_files, config['max_samples']
    )

    if not memory_ok:
        print(" text,textrunningtraining")
        print(" suggestion:")
        print("  1. textmax_samplestextnumber of samplestext")
        print("  2. decreasebatch_size")
        print("  3. disabledtextexpert")
        print("  4. textsystem memory")
        return


    os.makedirs(config['model_save_dir'], exist_ok=True)
    save_path = os.path.join(config['model_save_dir'], config['model_save_name'])

    model_class, model_kwargs = create_heterogeneous_model_config(
        config['model_config']
    )

    print("\n" + "="*60)
    print(" heterogeneous experttextfeatures:")
    print("   image expert: CNNtext,textfeatures")
    print("   text expert: text,textfeatures")
    print("   IDexpert: text,textIDfeatures")
    print("   graph expert: LightGCN,text")
    print("   textMLPexpert: Wide&Deep,textfeaturestext")
    print("   textexperttext")

    print(" text:")
    print("   textimprovement: textexpertAUCimprovement2-5%")
    print("   interpretability: experttextfeaturestextanalysis")
    print("   robustness: cold starttextexpertcompensation")
    print("   flexibility: can be optimized independentlytextexperthyperparameters")

    print("="*60)

    try:

        print(f"\n textheterogeneous experttraining...")

        best_auc = train_model_memory_single_gpu_full_embeddings(
            model_class=model_class,
            model_kwargs=model_kwargs,
            train_files=train_files,
            test_files=test_files,
            epochs=config['epochs'],
            lr=config['learning_rate'],
            batch_size=config['batch_size'],
            save_path=save_path,
            early_stop=config['early_stop'],
            max_samples=config['max_samples'],
            use_mmap=config['use_mmap']
        )

        print(f"\n heterogeneous experttraining completed!")
        print("=" * 90)
        print(f" best AUC: {best_auc:.4f}")
        print(f" model saved: {save_path}")


        if os.path.exists(save_path):
            model_size = os.path.getsize(save_path) / (1024**2)
            print(f" model file size: {model_size:.1f} MB")

            try:
                checkpoint = torch.load(save_path, map_location='cpu')
                print(" model validation passed")
                print(f"  training epochs: {checkpoint['epoch']}")
                print(f"  final AUC: {checkpoint['test_auc']:.4f}")
                print(f"   streaming: {checkpoint.get('use_streaming', 'text')}")
                print(f"   heterogeneous expert: {checkpoint.get('use_lightgcn', 'text')}")


                print(f"\n textrunningexperttextanalysis...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


                model = model_class(**model_kwargs).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])


                analyze_expert_performance(model, test_files, device)

            except Exception as e:
                print(f" model validation failed: {e}")


        memory_info = psutil.virtual_memory()
        print(f"\n text:")
        print(f"  memory usage: {memory_info.percent:.1f}%")
        print(f"  trainingtext: heterogeneous expertstreaming")

        print(f"\n heterogeneous experttrainingsummary:")
        print(f"   final AUC: {best_auc:.4f}")
        print(f"   architecturetext: 5heterogeneous experttext")
        print(f"   trainingtext: streaming,text")
        print(f"   textscene: texthigh-accuracyandinterpretabilitytextrecommender system")
        print(f"   analysistext: experttextfeaturestextanalysis")

    except Exception as e:
        print(f"\n trainingtext: {e}")
        import traceback
        traceback.print_exc()

        print(f"\n textsuggestion:")
        print(f"  - check whether data files are complete")
        print(f"  - reducemax_samplesparameters")
        print(f"  - decreasebatch_size")
        print(f"  - checkingtextexperttextparameterstext")

        if config['model_config']['use_lightgcn']:
            print(f"   LightGCNtext:")
            print(f"    - checkinggcn_logs_incrementaldirectory")
            print(f"    - checkingscenegraph embeddingsfile")
            print(f"    - trydisabledLightGCN graph expert")

        return

    print("\n heterogeneous expertmultimodalMMoEtraining completed!")
    print(f" textexperttextfeaturestext!")
    print(f" heterogeneousarchitecturetexteverytextfeaturestexttotext!")
    print(f" texthigh-accuracy,high-interpretabilitytext!")


def quick_heterogeneous_test(data_files=None, max_samples=1000):
    print(f" textheterogeneous experttexttesting...")

    if data_files is None:
        data_files = [
            "/root/text/baseline/embeddings/antm2c_10m_part0_train_embeddings.h5"
        ]

    try:

        data = load_single_h5_file_to_memory_full_embeddings(
            data_files[0],
            max_samples=max_samples
        )

        print(f" data loadingsucceeded!")
        print(f"  number of samples: {len(data['labels']):,}")


        model = create_heterogeneous_mmoe_model()


        batch_size = min(10, len(data['labels']))
        image_emb = torch.tensor(data['image_embeddings'][:batch_size], dtype=torch.float32)
        text_emb = torch.tensor(data['text_embeddings'][:batch_size], dtype=torch.float32)
        id_emb = torch.tensor(data['id_embeddings'][:batch_size], dtype=torch.float32)
        scenes = torch.tensor(data['scenes'][:batch_size], dtype=torch.long)
        user_ids = data['user_ids'][:batch_size]
        item_ids = data['item_ids'][:batch_size]


        model.eval()
        with torch.no_grad():
            output = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)

        print(f" heterogeneous experttexttestingsucceeded!")
        print(f"  output shape: {output.shape}")
        print(f"  output range: [{output.min():.3f}, {output.max():.3f}]")


        expert_weights = model.get_expert_weights(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
        print(f"  expert weightsanalysis: ")

        contributions = model.get_expert_contributions(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
        print(f"  expert contributionanalysis: ")

        return True

    except Exception as e:
        print(f" testingfailed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "test":

            max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

            print(f" heterogeneous experttestingparameters:")
            print(f"  number of samples: {max_samples}")
            print(f"   testingheterogeneous expertarchitecture")

            quick_heterogeneous_test(max_samples=max_samples)

        elif mode == "train":

            main()

        else:
            print(" unknown mode,supported modes: test, train")
            print(" usage:")
            print("  python train_main_heterogeneous.py train                    # startingheterogeneous experttraining")
            print("  python train_main_heterogeneous.py test [samples]           # testingheterogeneous experttext")
            print("    example: python train_main_heterogeneous.py test 1000        # testing1000text")
            print("   features: 5heterogeneous experttext,textandinterpretabilitytextimprovement")
    else:

        print(" textheterogeneous experttextGPUtraining...")
        main()