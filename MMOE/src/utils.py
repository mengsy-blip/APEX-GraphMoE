# utils_single_gpu_no_lightgcn.py - 单GPU版本工具文件（集成LightGCN图专家，使用全嵌入）
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
    """基于H5文件的数据集类，使用所有ID、文本和图像嵌入（支持LightGCN图专家）"""
    def __init__(self, h5_files, max_samples_per_file=None):
        """
        初始化H5数据集（使用全嵌入+LightGCN图专家）
        
        Args:
            h5_files: H5文件路径列表
            max_samples_per_file: 每个文件最大样本数限制
        """
        self.h5_files = h5_files if isinstance(h5_files, list) else [h5_files]
        self.file_sample_counts = []
        self.cumulative_counts = [0]
        
        # 计算每个文件的样本数
        print("🔄 分析H5文件（全嵌入模式+LightGCN图专家）...")
        for i, h5_file in enumerate(self.h5_files):
            print(f"  处理文件 [{i+1}/{len(self.h5_files)}]: {os.path.basename(h5_file)}")
            try:
                with h5py.File(h5_file, 'r') as f:
                    num_samples = f['labels'].shape[0]
                    if max_samples_per_file:
                        num_samples = min(num_samples, max_samples_per_file)
                    self.file_sample_counts.append(num_samples)
                    self.cumulative_counts.append(self.cumulative_counts[-1] + num_samples)
                    print(f"    样本数: {num_samples:,}")
            except Exception as e:
                print(f"    ⚠️ 文件读取错误: {e}")
                self.file_sample_counts.append(0)
                self.cumulative_counts.append(self.cumulative_counts[-1])
        
        self.total_samples = self.cumulative_counts[-1]
        print(f"✅ 数据集初始化完成（全嵌入模式+LightGCN图专家）: {len(self.h5_files)} 个文件, 共 {self.total_samples:,} 个样本")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")
        
        # 找到对应的文件和本地索引
        file_idx = 0
        while file_idx < len(self.cumulative_counts) - 1:
            if idx < self.cumulative_counts[file_idx + 1]:
                break
            file_idx += 1
        
        local_idx = idx - self.cumulative_counts[file_idx]
        
        # 从对应文件读取数据
        try:
            with h5py.File(self.h5_files[file_idx], 'r') as f:
                sample = {}
                
                # 读取图像嵌入 (256维)
                if 'image_embeddings' in f:
                    sample['image_embeddings'] = torch.tensor(f['image_embeddings'][local_idx], dtype=torch.float32)
                else:
                    sample['image_embeddings'] = torch.zeros(256, dtype=torch.float32)
                
                # 读取所有文本嵌入 (各768维)
                text_fields = ['item_title', 'item_entity_names', 'bill_entity_seq', 'service_entity_seq', 'query_entity_seq']
                text_embeddings = []
                for field in text_fields:
                    field_key = f'{field}_embeddings'
                    if field_key in f:
                        text_embeddings.append(f[field_key][local_idx])
                    else:
                        text_embeddings.append(np.zeros(768, dtype=np.float32))
                
                # 合并所有文本嵌入 (5*768=3840维)
                sample['text_embeddings'] = torch.tensor(np.concatenate(text_embeddings), dtype=torch.float32)
                
                # 读取所有ID嵌入
                id_embeddings = []
                
                # 用户和物品ID嵌入
                for id_field in ['user_id', 'item_id']:
                    field_key = f'{id_field}_embeddings'
                    if field_key in f:
                        id_embeddings.append(f[field_key][local_idx])
                    else:
                        id_embeddings.append(np.zeros(32, dtype=np.float32))
                
                # 深度特征ID嵌入
                for i in range(27):
                    field_key = f'deep_features_{i}_embeddings'
                    if field_key in f:
                        id_embeddings.append(f[field_key][local_idx])
                    else:
                        id_embeddings.append(np.zeros(32, dtype=np.float32))
                
                # 合并所有ID嵌入 (29*32=928维)
                sample['id_embeddings'] = torch.tensor(np.concatenate(id_embeddings), dtype=torch.float32)
                
                # 读取标签
                sample['label'] = torch.tensor(f['labels'][local_idx], dtype=torch.float32)
                
                # 读取元数据
                metadata = f['metadata'][local_idx]
                user_id_str = metadata[0].decode('utf-8') if isinstance(metadata[0], bytes) else str(metadata[0])
                item_id_str = metadata[1].decode('utf-8') if isinstance(metadata[1], bytes) else str(metadata[1])
                scene_str = metadata[2].decode('utf-8') if isinstance(metadata[2], bytes) else str(metadata[2])
                
                # 🔥 新增：用户ID和物品ID（LightGCN图专家需要）
                sample['user_id'] = user_id_str
                sample['item_id'] = item_id_str
                
                # 处理场景ID
                try:
                    scene_id = int(scene_str) if scene_str.isdigit() else 0
                except:
                    scene_id = 0
                sample['scene'] = torch.tensor(scene_id, dtype=torch.long)
                
                return sample
                
        except Exception as e:
            print(f"⚠️ 读取样本 {idx} 时出错: {e}")
            # 返回默认样本
            return self._get_default_sample()

    def _get_default_sample(self):
        """返回默认样本（用于出错时的fallback）"""
        return {
            'image_embeddings': torch.zeros(256, dtype=torch.float32),
            'text_embeddings': torch.zeros(3840, dtype=torch.float32),  # 5*768
            'id_embeddings': torch.zeros(928, dtype=torch.float32),     # 29*32
            'label': torch.tensor(0.0, dtype=torch.float32),
            'scene': torch.tensor(0, dtype=torch.long),
            'user_id': "default_user",  # 🔥 新增
            'item_id': "default_item",  # 🔥 新增
        }


def find_h5_files(data_dir, pattern="*_embeddings.h5"):
    """
    在指定目录中查找H5文件
    
    Args:
        data_dir: 数据目录
        pattern: 文件匹配模式
        
    Returns:
        list: H5文件路径列表
    """
    if not os.path.exists(data_dir):
        print(f"⚠️ 目录不存在: {data_dir}")
        return []
    
    h5_files = glob.glob(os.path.join(data_dir, pattern))
    h5_files.sort()  # 排序以保证一致性
    
    print(f"📁 在 {data_dir} 中找到 {len(h5_files)} 个H5文件:")
    for i, file_path in enumerate(h5_files):
        file_size = os.path.getsize(file_path) / (1024**3)  # GB
        print(f"  [{i+1}] {os.path.basename(file_path)} ({file_size:.2f} GB)")
    
    return h5_files


def analyze_h5_file_full_embeddings(h5_file_path):
    """
    分析单个H5文件的结构和内容（全嵌入模式+LightGCN图专家）
    
    Args:
        h5_file_path: H5文件路径
    """
    print(f"🔍 分析文件（全嵌入模式+LightGCN图专家）: {os.path.basename(h5_file_path)}")
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            print(f"📊 文件属性:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            
            print(f"\n📊 数据集:")
            
            # 统计各类嵌入维度
            image_dim = 0
            text_dims = {}
            id_dims = {}
            
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    shape = f[key].shape
                    dtype = f[key].dtype
                    size_mb = f[key].nbytes / (1024**2)
                    
                    print(f"  📦 {key}: {shape} ({dtype}) - {size_mb:.1f} MB")
                    
                    # 分类统计
                    if key == 'image_embeddings':
                        image_dim = shape[1] if len(shape) > 1 else 0
                    elif key.endswith('_embeddings') and any(text_field in key for text_field in ['item_title', 'item_entity_names', 'bill_entity_seq', 'service_entity_seq', 'query_entity_seq']):
                        field_name = key.replace('_embeddings', '')
                        text_dims[field_name] = shape[1] if len(shape) > 1 else 0
                    elif key.endswith('_embeddings') and ('user_id' in key or 'item_id' in key or 'deep_features' in key):
                        field_name = key.replace('_embeddings', '')
                        id_dims[field_name] = shape[1] if len(shape) > 1 else 0
                        
                elif isinstance(f[key], h5py.Group):
                    print(f"  📁 {key}: Group with {len(f[key])} items")
            
            print(f"\n✅ 特征维度统计:")
            print(f"  🖼️ 图像嵌入: {image_dim}维")
            
            total_text_dim = sum(text_dims.values())
            print(f"  📝 文本嵌入: {total_text_dim}维 (共{len(text_dims)}个字段)")
            for field, dim in text_dims.items():
                print(f"    - {field}: {dim}维")
            
            total_id_dim = sum(id_dims.values())
            print(f"  🆔 ID嵌入: {total_id_dim}维 (共{len(id_dims)}个特征)")
            for field, dim in sorted(id_dims.items()):
                print(f"    - {field}: {dim}维")
            
            print(f"  🎯 总特征维度: {image_dim + total_text_dim + total_id_dim}维")
            print(f"  🔥 + LightGCN图嵌入: 动态维度（按场景加载）")
            
            # 检查样本示例
            if 'metadata' in f and f['metadata'].shape[0] > 0:
                print(f"\n🔍 样本示例 (前3个):")
                for i in range(min(3, f['metadata'].shape[0])):
                    metadata = f['metadata'][i]
                    user_id = metadata[0].decode('utf-8') if isinstance(metadata[0], bytes) else str(metadata[0])
                    item_id = metadata[1].decode('utf-8') if isinstance(metadata[1], bytes) else str(metadata[1])
                    scene = metadata[2].decode('utf-8') if isinstance(metadata[2], bytes) else str(metadata[2])
                    label = f['labels'][i] if 'labels' in f else 'N/A'
                    print(f"  样本 {i}: user_id={user_id}, item_id={item_id}, scene={scene}, label={label}")
                    print(f"    🔥 LightGCN将使用user_id和item_id查找图嵌入")
                    
    except Exception as e:
        print(f"❌ 分析文件时出错: {e}")


def check_lightgcn_embeddings(gcn_logs_dir="gcn_logs_incremental"):
    """
    检查LightGCN图嵌入文件
    
    Args:
        gcn_logs_dir: LightGCN模型保存目录
        
    Returns:
        dict: 检查结果
    """
    print(f"🔥 检查LightGCN图嵌入文件...")
    print(f"  目录: {gcn_logs_dir}")
    
    result = {
        'available': False,
        'scenes': {},
        'total_size_gb': 0,
        'error_msg': None
    }
    
    if not os.path.exists(gcn_logs_dir):
        result['error_msg'] = f"LightGCN目录不存在: {gcn_logs_dir}"
        print(f"❌ {result['error_msg']}")
        return result
    
    try:
        # 查找所有场景目录
        scene_dirs = [d for d in os.listdir(gcn_logs_dir) if d.startswith('scene_')]
        scene_dirs.sort()
        
        print(f"  发现场景目录: {len(scene_dirs)} 个")
        
        for scene_dir in scene_dirs:
            scene_path = os.path.join(gcn_logs_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            
            try:
                scene_id = int(scene_dir.split('_')[1])
            except (IndexError, ValueError):
                print(f"    ⚠️ 场景目录格式错误: {scene_dir}")
                continue
            
            # 检查用户和物品嵌入文件
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
                    user_emb = np.load(user_emb_path, mmap_mode='r')  # 只读取形状
                    scene_info['user_shape'] = user_emb.shape
                    scene_info['emb_dim'] = user_emb.shape[1] if len(user_emb.shape) > 1 else 0
                except Exception as e:
                    print(f"    ⚠️ 读取用户嵌入失败: {e}")
            
            if scene_info['item_emb_available']:
                scene_info['item_emb_size_gb'] = os.path.getsize(item_emb_path) / (1024**3)
                try:
                    item_emb = np.load(item_emb_path, mmap_mode='r')  # 只读取形状
                    scene_info['item_shape'] = item_emb.shape
                    if scene_info['emb_dim'] == 0:  # 如果用户嵌入读取失败
                        scene_info['emb_dim'] = item_emb.shape[1] if len(item_emb.shape) > 1 else 0
                except Exception as e:
                    print(f"    ⚠️ 读取物品嵌入失败: {e}")
            
            # 检查ID映射文件
            mapping_files = ['user_id_to_idx.pkl', 'item_id_to_idx.pkl', 'idx_to_user_id.pkl', 'idx_to_item_id.pkl']
            scene_info['mapping_files'] = {}
            for mapping_file in mapping_files:
                mapping_path = os.path.join(scene_path, mapping_file)
                scene_info['mapping_files'][mapping_file] = os.path.exists(mapping_path)
            
            result['scenes'][scene_id] = scene_info
            result['total_size_gb'] += scene_info['user_emb_size_gb'] + scene_info['item_emb_size_gb']
            
            # 打印场景信息
            status = "✅" if (scene_info['user_emb_available'] and scene_info['item_emb_available']) else "⚠️"
            print(f"    {status} 场景{scene_id}:")
            if scene_info['user_emb_available']:
                print(f"      用户嵌入: {scene_info['user_shape']} ({scene_info['user_emb_size_gb']:.3f} GB)")
            else:
                print(f"      用户嵌入: 缺失")
            
            if scene_info['item_emb_available']:
                print(f"      物品嵌入: {scene_info['item_shape']} ({scene_info['item_emb_size_gb']:.3f} GB)")
            else:
                print(f"      物品嵌入: 缺失")
            
            if scene_info['emb_dim'] > 0:
                print(f"      嵌入维度: {scene_info['emb_dim']}")
            
            # 检查映射文件
            available_mappings = sum(scene_info['mapping_files'].values())
            print(f"      ID映射文件: {available_mappings}/{len(mapping_files)} 个可用")
        
        # 汇总结果
        available_scenes = [s for s, info in result['scenes'].items() 
                          if info['user_emb_available'] and info['item_emb_available']]
        
        if available_scenes:
            result['available'] = True
            print(f"\n✅ LightGCN图嵌入检查完成:")
            print(f"  可用场景: {len(available_scenes)} 个 {available_scenes}")
            print(f"  总大小: {result['total_size_gb']:.3f} GB")
        else:
            result['error_msg'] = "没有找到完整的场景图嵌入"
            print(f"\n⚠️ {result['error_msg']}")
        
    except Exception as e:
        result['error_msg'] = f"检查LightGCN图嵌入时出错: {e}"
        print(f"❌ {result['error_msg']}")
    
    return result


def split_h5_files(h5_files, train_ratio=0.8):
    """
    将H5文件列表分割为训练集和测试集
    
    Args:
        h5_files: H5文件路径列表
        train_ratio: 训练集比例
        
    Returns:
        tuple: (train_files, test_files)
    """
    # 根据文件名自动分割
    train_files = []
    test_files = []
    
    for h5_file in h5_files:
        file_name = os.path.basename(h5_file).lower()
        if 'train' in file_name:
            train_files.append(h5_file)
        elif 'test' in file_name:
            test_files.append(h5_file)
        else:
            # 如果文件名中没有明确标识，按比例分配
            if len(train_files) < len(h5_files) * train_ratio:
                train_files.append(h5_file)
            else:
                test_files.append(h5_file)
    
    print(f"📂 数据集分割:")
    print(f"  训练文件: {len(train_files)} 个")
    for f in train_files:
        print(f"    - {os.path.basename(f)}")
    print(f"  测试文件: {len(test_files)} 个")
    for f in test_files:
        print(f"    - {os.path.basename(f)}")
    
    return train_files, test_files


def get_h5_stats_full_embeddings(h5_files):
    """
    获取H5文件的统计信息（全嵌入模式+LightGCN图专家）
    
    Args:
        h5_files: H5文件路径列表
        
    Returns:
        dict: 统计信息
    """
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
            'graph': 'dynamic'  # 🔥 LightGCN图嵌入维度动态
        },
        'total_feature_dim': 256 + 3840 + 928,  # 5024维（不包含动态图嵌入）
        'lightgcn_info': None  # 🔥 新增LightGCN信息
    }
    
    print("🔄 统计H5文件信息（全嵌入模式+LightGCN图专家）...")
    
    for h5_file in tqdm(h5_files, desc="处理文件"):
        try:
            file_size_gb = os.path.getsize(h5_file) / (1024**3)
            stats['total_size_gb'] += file_size_gb
            
            with h5py.File(h5_file, 'r') as f:
                # 从属性获取统计信息
                if 'num_samples' in f.attrs:
                    stats['total_samples'] += f.attrs['num_samples']
                else:
                    # 如果没有属性，从数据集形状获取
                    if 'labels' in f:
                        stats['total_samples'] += f['labels'].shape[0]
                
                if 'positive_samples' in f.attrs:
                    stats['positive_samples'] += f.attrs['positive_samples']
                
                if 'negative_samples' in f.attrs:
                    stats['negative_samples'] += f.attrs['negative_samples']
                    
        except Exception as e:
            print(f"  ⚠️ 处理文件 {h5_file} 时出错: {e}")
            continue
    
    # 计算正负样本比例
    total_labeled = stats['positive_samples'] + stats['negative_samples']
    if total_labeled > 0:
        stats['positive_ratio'] = stats['positive_samples'] / total_labeled
        stats['negative_ratio'] = stats['negative_samples'] / total_labeled
    else:
        stats['positive_ratio'] = 0.0
        stats['negative_ratio'] = 0.0
    
    # 🔥 新增：检查LightGCN图嵌入信息
    lightgcn_info = check_lightgcn_embeddings()
    stats['lightgcn_info'] = lightgcn_info
    if lightgcn_info['available']:
        stats['total_size_gb'] += lightgcn_info['total_size_gb']
        
        # 计算图嵌入维度
        first_scene = next(iter(lightgcn_info['scenes'].values()), {})
        if first_scene.get('emb_dim', 0) > 0:
            stats['feature_dimensions']['graph'] = first_scene['emb_dim']
    
    return stats


def print_training_recommendations_full_embeddings(stats):
    """
    根据数据统计打印训练建议（全嵌入模式+LightGCN图专家）
    
    Args:
        stats: 从get_h5_stats_full_embeddings返回的统计信息
    """
    print("\n💡 单GPU训练建议（全嵌入模式+LightGCN图专家）:")
    
    # 基于数据量的建议
    if stats['total_samples'] > 5000000:  # 500万以上
        print("  📊 大数据量 (>5M 样本):")
        print("    - batch_size: 512-1024 (特征维度较高)")
        print("    - learning_rate: 3e-5 - 5e-5")
        print("    - 使用mixed precision训练")
        print("    - 考虑梯度累积")
        print("    - 🔥 LightGCN图专家可能消耗额外内存")
    elif stats['total_samples'] > 1000000:  # 100万以上
        print("  📊 中等数据量 (1M-5M 样本):")
        print("    - batch_size: 1024-2048") 
        print("    - learning_rate: 5e-5 - 1e-4")
        print("    - 标准训练配置")
        print("    - 🔥 LightGCN图专家性能提升明显")
    else:
        print("  📊 小数据量 (<1M 样本):")
        print("    - batch_size: 256-512")
        print("    - learning_rate: 1e-4 - 2e-4")
        print("    - 可能需要更多epochs")
        print("    - 🔥 LightGCN图专家有助于缓解数据稀疏")
    
    # 强调特征维度的影响
    total_dim = stats['total_feature_dim']
    print(f"\n✨ 特征维度分析:")
    print(f"    - 传统特征维度: {total_dim}维")
    print(f"    - 图像特征: {stats['feature_dimensions']['image']}维")
    print(f"    - 文本特征: {stats['feature_dimensions']['text']}维")
    print(f"    - ID特征: {stats['feature_dimensions']['id']}维")
    
    # 🔥 LightGCN图专家分析
    lightgcn_info = stats.get('lightgcn_info')
    if lightgcn_info and lightgcn_info['available']:
        graph_dim = stats['feature_dimensions'].get('graph', 'dynamic')
        print(f"    - 🔥 LightGCN图特征: {graph_dim}维")
        print(f"    - 🔥 可用场景: {len(lightgcn_info['scenes'])} 个")
        print(f"    - 🔥 图嵌入大小: {lightgcn_info['total_size_gb']:.3f} GB")
        print(f"    - 特征丰富度: 极高 (多模态+多文本+多ID+图结构)")
    else:
        print(f"    - ⚠️ LightGCN图专家: 不可用")
        print(f"    - 特征丰富度: 高 (多模态+多文本+多ID)")
        if lightgcn_info and lightgcn_info.get('error_msg'):
            print(f"    - 错误信息: {lightgcn_info['error_msg']}")
    
    # 内存和计算建议
    base_memory_estimate = (stats['total_samples'] * total_dim * 4) / (1024**3)  # 4 bytes per float32
    lightgcn_memory = lightgcn_info['total_size_gb'] if (lightgcn_info and lightgcn_info['available']) else 0
    
    print(f"\n💾 内存预估:")
    print(f"    - 传统特征数据: ~{base_memory_estimate:.2f} GB")
    if lightgcn_memory > 0:
        print(f"    - 🔥 LightGCN图嵌入: ~{lightgcn_memory:.3f} GB")
        print(f"    - 总内存需求: ~{base_memory_estimate + lightgcn_memory:.2f} GB")
    else:
        print(f"    - 总内存需求: ~{base_memory_estimate:.2f} GB")
    print(f"    - 建议使用FP16降低内存使用")
    print(f"    - 考虑内存映射技术")
    
    # 基于文件大小的建议
    if stats['total_size_gb'] > 20:
        print("  💾 大文件 (>20GB):")
        print("    - 增加num_workers (4-8)")
        print("    - 使用pin_memory=True")
        print("    - 考虑数据预加载")
        print("    - 🔥 启用内存映射模式")
    
    # 基于样本分布的建议
    if 'positive_ratio' in stats and stats['positive_ratio'] > 0:
        if stats['positive_ratio'] < 0.1 or stats['positive_ratio'] > 0.9:
            print("  ⚖️ 样本不平衡:")
            print("    - 考虑使用加权损失函数")
            print("    - 调整positive/negative采样比例")
            print(f"    - 当前正样本比例: {stats['positive_ratio']:.2%}")
            print("    - 🔥 LightGCN图专家有助于处理不平衡数据")
    
    # 模型建议
    print(f"\n🎯 模型架构建议:")
    print(f"    - 使用多专家网络处理不同模态")
    print(f"    - 图像专家: CNN/Transformer")
    print(f"    - 文本专家: 注意力机制")
    print(f"    - ID专家: MLP/Embedding")
    
    if lightgcn_info and lightgcn_info['available']:
        print(f"    - 🔥 LightGCN图专家: 处理用户-物品交互图")
        print(f"    - 🔥 图专家优势: 捕获协同过滤信息")
        print(f"    - 🔥 融合策略: 门控机制自动权衡各专家")
        print(f"    - 🔥 预期提升: AUC +2~5%")
    else:
        print(f"    - ⚠️ LightGCN图专家不可用，建议:")
        print(f"      • 检查gcn_logs_incremental目录")
        print(f"      • 运行LightGCN预训练步骤")
        print(f"      • 或禁用图专家 (use_lightgcn=False)")
    
    print(f"    - 融合层: 注意力加权或门控机制")


def quick_h5_check_full_embeddings(data_dir_or_files):
    """
    快速检查H5数据（全嵌入模式+LightGCN图专家）
    
    Args:
        data_dir_or_files: 数据目录路径或文件列表
        
    Returns:
        bool: 检查是否通过
    """
    if isinstance(data_dir_or_files, str):
        h5_files = find_h5_files(data_dir_or_files)
    else:
        h5_files = data_dir_or_files
    
    if not h5_files:
        print("❌ 未找到H5文件")
        return False
    
    # 获取统计信息
    stats = get_h5_stats_full_embeddings(h5_files[:3])  # 只检查前3个文件
    
    print(f"\n📊 数据概览（全嵌入模式+LightGCN图专家）:")
    print(f"  文件数量: {len(h5_files)}")
    print(f"  样本数量: {stats['total_samples']:,}")
    print(f"  数据大小: {stats['total_size_gb']:.2f} GB")
    print(f"  传统特征维度: {stats['total_feature_dim']}维")
    
    # 🔥 LightGCN图专家检查结果
    lightgcn_info = stats.get('lightgcn_info')
    if lightgcn_info:
        if lightgcn_info['available']:
            print(f"  🔥 LightGCN图专家: ✅ 可用")
            print(f"    - 可用场景: {len(lightgcn_info['scenes'])} 个")
            print(f"    - 图嵌入大小: {lightgcn_info['total_size_gb']:.3f} GB")
            if lightgcn_info['scenes']:
                first_scene = next(iter(lightgcn_info['scenes'].values()))
                if first_scene.get('emb_dim', 0) > 0:
                    print(f"    - 图嵌入维度: {first_scene['emb_dim']}维")
        else:
            print(f"  🔥 LightGCN图专家: ⚠️ 不可用")
            if lightgcn_info.get('error_msg'):
                print(f"    - 原因: {lightgcn_info['error_msg']}")
    
    if stats['positive_samples'] > 0:
        print(f"  正样本: {stats['positive_samples']:,} ({stats['positive_ratio']:.2%})")
        print(f"  负样本: {stats['negative_samples']:,} ({stats['negative_ratio']:.2%})")
    
    # 打印训练建议
    print_training_recommendations_full_embeddings(stats)
    
    return True


def create_h5_dataset_full_embeddings(h5_files, **kwargs):
    """
    创建全嵌入H5数据集（支持LightGCN图专家）
    
    Args:
        h5_files: H5文件路径列表
        **kwargs: 其他H5Dataset参数
        
    Returns:
        H5Dataset_FullEmbeddings实例
    """
    return H5Dataset_FullEmbeddings(
        h5_files=h5_files,
        **kwargs
    )


def estimate_feature_importance():
    """
    估算不同特征的重要性权重（包含LightGCN图专家）
    
    Returns:
        dict: 特征重要性权重
    """
    weights = {
        'image_weight': 0.25,     # 图像特征权重
        'text_weight': 0.4,       # 文本特征权重 (多个文本字段)
        'id_weight': 0.15,        # ID特征权重 (用户/物品/深度特征)
        'graph_weight': 0.2,      # 🔥 LightGCN图特征权重
        'fusion_strategy': 'mmoe_gating'  # 推荐使用MMoE门控机制融合
    }
    
    return weights


def compare_feature_usage(num_samples, feature_dims):
    """
    比较不同特征组合的内存和计算成本（包含LightGCN图专家）
    
    Args:
        num_samples: 样本数量
        feature_dims: 特征维度字典
    """
    scenarios = {
        '仅图像': feature_dims['image'],
        '仅文本': feature_dims['text'], 
        '仅ID': feature_dims['id'],
        '图像+文本': feature_dims['image'] + feature_dims['text'],
        '传统全特征': feature_dims['image'] + feature_dims['text'] + feature_dims['id'],
        '🔥 全特征+LightGCN': feature_dims['image'] + feature_dims['text'] + feature_dims['id']  # 图嵌入动态
    }
    
    print(f"\n📊 特征组合对比 ({num_samples:,} 样本):")
    for scenario, dim in scenarios.items():
        if 'LightGCN' in scenario:
            memory_gb = (num_samples * dim * 4) / (1024**3)  # 传统特征
            print(f"  {scenario}: {dim}维 + 动态图嵌入 -> {memory_gb:.2f} GB + 图嵌入")
        else:
            memory_gb = (num_samples * dim * 4) / (1024**3)  # 4 bytes per float32
            print(f"  {scenario}: {dim}维 -> {memory_gb:.2f} GB")
    
    print(f"\n💡 建议:")
    print(f"  - 从简单特征组合开始实验")
    print(f"  - 逐步增加特征复杂度")
    print(f"  - 使用特征重要性分析优化")
    print(f"  - 🔥 LightGCN图专家通常带来显著提升")
    print(f"  - 🔥 如内存不足，可先禁用图专家")


def validate_lightgcn_compatibility(h5_files, gcn_logs_dir="gcn_logs_incremental"):
    """
    验证H5数据与LightGCN图嵌入的兼容性
    
    Args:
        h5_files: H5文件路径列表
        gcn_logs_dir: LightGCN模型保存目录
        
    Returns:
        dict: 兼容性检查结果
    """
    print(f"🔥 验证H5数据与LightGCN图嵌入兼容性...")
    
    result = {
        'compatible': False,
        'h5_scenes': set(),
        'lightgcn_scenes': set(),
        'missing_scenes': set(),
        'sample_user_ids': [],
        'sample_item_ids': [],
        'compatibility_issues': []
    }
    
    # 1. 检查H5文件中的场景和ID
    print("  检查H5文件中的场景和用户/物品ID...")
    for h5_file in h5_files[:2]:  # 只检查前2个文件
        if not os.path.exists(h5_file):
            continue
            
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'metadata' not in f:
                    result['compatibility_issues'].append(f"文件缺少metadata: {h5_file}")
                    continue
                
                # 采样检查前100个样本
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
                        result['compatibility_issues'].append(f"无效场景ID: {scene_str}")
                    
                    if len(result['sample_user_ids']) < 10:
                        result['sample_user_ids'].append(user_id)
                    if len(result['sample_item_ids']) < 10:
                        result['sample_item_ids'].append(item_id)
                        
        except Exception as e:
            result['compatibility_issues'].append(f"读取H5文件失败 {h5_file}: {e}")
    
    # 2. 检查LightGCN图嵌入
    lightgcn_info = check_lightgcn_embeddings(gcn_logs_dir)
    if lightgcn_info['available']:
        result['lightgcn_scenes'] = set(lightgcn_info['scenes'].keys())
        
        # 3. 检查场景兼容性
        result['missing_scenes'] = result['h5_scenes'] - result['lightgcn_scenes']
        
        if result['missing_scenes']:
            result['compatibility_issues'].append(f"缺少场景图嵌入: {result['missing_scenes']}")
        
        # 4. 检查ID映射（采样检查）
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
                    
                    # 检查样本ID是否在映射中
                    missing_users = [uid for uid in result['sample_user_ids'][:5] if uid not in user_mapping]
                    missing_items = [iid for iid in result['sample_item_ids'][:5] if iid not in item_mapping]
                    
                    if missing_users:
                        result['compatibility_issues'].append(f"场景{scene_id}缺少用户ID映射: {missing_users[:3]}...")
                    if missing_items:
                        result['compatibility_issues'].append(f"场景{scene_id}缺少物品ID映射: {missing_items[:3]}...")
                        
                except Exception as e:
                    result['compatibility_issues'].append(f"场景{scene_id}读取ID映射失败: {e}")
            else:
                result['compatibility_issues'].append(f"场景{scene_id}缺少ID映射文件")
        
        # 5. 判断整体兼容性
        common_scenes = result['h5_scenes'] & result['lightgcn_scenes']
        if common_scenes and len(result['compatibility_issues']) == 0:
            result['compatible'] = True
        elif common_scenes:
            result['compatible'] = True  # 有部分兼容
    else:
        result['compatibility_issues'].append("LightGCN图嵌入不可用")
    
    # 打印检查结果
    print(f"\n📊 兼容性检查结果:")
    print(f"  H5文件场景: {sorted(result['h5_scenes'])}")
    print(f"  LightGCN场景: {sorted(result['lightgcn_scenes'])}")
    if result['missing_scenes']:
        print(f"  ⚠️ 缺失场景: {sorted(result['missing_scenes'])}")
    
    if result['compatible']:
        print(f"  ✅ 兼容性: 良好")
        if result['compatibility_issues']:
            print(f"  ⚠️ 警告: {len(result['compatibility_issues'])} 个问题")
            for issue in result['compatibility_issues'][:3]:
                print(f"    - {issue}")
    else:
        print(f"  ❌ 兼容性: 存在问题")
        for issue in result['compatibility_issues']:
            print(f"    - {issue}")
    
    return result


# 批量处理函数
def batch_analyze_h5_files_full_embeddings(h5_files, max_files=5):
    """
    批量分析H5文件（全嵌入模式+LightGCN图专家）
    
    Args:
        h5_files: H5文件路径列表
        max_files: 最大分析文件数
    """
    print(f"🔍 批量分析H5文件（全嵌入模式+LightGCN图专家，最多{max_files}个）...")
    
    files_to_analyze = h5_files[:max_files]
    
    for i, h5_file in enumerate(files_to_analyze):
        print(f"\n{'='*50}")
        print(f"分析文件 {i+1}/{len(files_to_analyze)}")
        analyze_h5_file_full_embeddings(h5_file)
    
    if len(h5_files) > max_files:
        print(f"\n⚠️ 还有 {len(h5_files) - max_files} 个文件未分析")
    
    print(f"\n📊 总体统计:")
    stats = get_h5_stats_full_embeddings(h5_files)
    print(f"  总文件数: {stats['total_files']}")
    print(f"  总样本数: {stats['total_samples']:,}")
    print(f"  总大小: {stats['total_size_gb']:.2f} GB")
    print(f"  传统特征维度: {stats['total_feature_dim']}维")
    
    # 🔥 LightGCN图专家总结
    lightgcn_info = stats.get('lightgcn_info')
    if lightgcn_info and lightgcn_info['available']:
        print(f"  🔥 LightGCN图专家: 可用 ({len(lightgcn_info['scenes'])}个场景)")
    else:
        print(f"  🔥 LightGCN图专家: 不可用")
    
    # 特征组合建议
    compare_feature_usage(stats['total_samples'], stats['feature_dimensions'])
    
    # 🔥 兼容性检查
    if h5_files:
        print(f"\n🔥 LightGCN兼容性检查:")
        compatibility = validate_lightgcn_compatibility(h5_files)
        if compatibility['compatible']:
            print("  ✅ H5数据与LightGCN图嵌入兼容")
        else:
            print("  ⚠️ 存在兼容性问题，建议检查数据")


# 🔥 新增：LightGCN专用工具函数
def prepare_lightgcn_for_training(gcn_logs_dir="gcn_logs_incremental", target_scenes=[0, 1, 2, 3, 4]):
    """
    为训练准备LightGCN图嵌入
    
    Args:
        gcn_logs_dir: LightGCN模型保存目录
        target_scenes: 目标场景列表
        
    Returns:
        bool: 准备是否成功
    """
    print(f"🔥 为训练准备LightGCN图嵌入...")
    print(f"  目录: {gcn_logs_dir}")
    print(f"  目标场景: {target_scenes}")
    
    if not os.path.exists(gcn_logs_dir):
        print(f"❌ LightGCN目录不存在: {gcn_logs_dir}")
        print("💡 请先运行LightGCN预训练步骤生成图嵌入")
        return False
    
    lightgcn_info = check_lightgcn_embeddings(gcn_logs_dir)
    
    if not lightgcn_info['available']:
        print(f"❌ LightGCN图嵌入不可用")
        if lightgcn_info.get('error_msg'):
            print(f"   原因: {lightgcn_info['error_msg']}")
        return False
    
    available_scenes = set(lightgcn_info['scenes'].keys())
    missing_scenes = set(target_scenes) - available_scenes
    
    if missing_scenes:
        print(f"⚠️ 缺少场景图嵌入: {sorted(missing_scenes)}")
        print(f"   可用场景: {sorted(available_scenes)}")
        print("💡 建议:")
        print("   1. 为缺失场景运行LightGCN预训练")
        print("   2. 或调整target_scenes参数")
        return False
    
    print(f"✅ LightGCN图嵌入准备完成")
    print(f"   可用场景: {sorted(available_scenes)}")
    print(f"   图嵌入大小: {lightgcn_info['total_size_gb']:.3f} GB")
    
    return True