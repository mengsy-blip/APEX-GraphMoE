# h5_train_memory_full_embeddings_topk.py - 基于流式训练的单GPU训练函数（全嵌入模式，支持LightGCN图专家+Top-K防极化）
# 🔄 改为使用第一套代码的流式数据读取方式 + Top-K防极化机制

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


# ============ 从第一套代码复制流式读取函数 ============

def load_single_h5_file_to_memory_full_embeddings(h5_file, max_samples=None):
    """
    🔧 简化版：直接按实际存在的ID特征读取
    """
    print(f"📂 加载单个文件: {os.path.basename(h5_file)}")
    
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"文件不存在: {h5_file}")
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # 🔧 第一步：检测所有数据集的样本数，找到最小值
            print(f"  🔍 检测维度一致性...")
            
            sample_counts = []
            dataset_info = {}
            
            # 检查所有相关数据集的样本数
            all_datasets = list(f.keys())
            
            for dataset_name in all_datasets:
                if isinstance(f[dataset_name], h5py.Dataset) and len(f[dataset_name].shape) > 0:
                    sample_count = f[dataset_name].shape[0]
                    sample_counts.append(sample_count)
                    dataset_info[dataset_name] = sample_count
            
            # 🔧 检测维度不一致问题
            if len(set(sample_counts)) > 1:
                min_samples = min(sample_counts)
                max_samples_found = max(sample_counts)
                
                print(f"  ⚠️ 检测到维度不一致问题:")
                print(f"    最小样本数: {min_samples:,}")
                print(f"    最大样本数: {max_samples_found:,}")
                print(f"    差异: {max_samples_found - min_samples:,} 样本")
                print(f"  🔧 自动修复: 截断到最小长度 {min_samples:,}")
                
                # 使用最小样本数
                num_samples = min_samples
            else:
                num_samples = sample_counts[0] if sample_counts else 0
                print(f"  ✅ 维度一致: {num_samples:,} 样本")
            
            # 如果设置了最大样本数限制，进一步限制
            if max_samples and num_samples > max_samples:
                num_samples = max_samples
                print(f"  🔧 用户限制: 截断到 {num_samples:,} 样本")
            
            print(f"  📊 最终样本数: {num_samples:,}")
            
            # 🔧 第二步：读取图像嵌入
            if 'image_embeddings' in f:
                image_emb = f['image_embeddings'][:num_samples]
                print(f"    图像嵌入: {image_emb.shape}")
            else:
                image_emb = np.zeros((num_samples, 256), dtype=np.float32)
                print(f"    图像嵌入: 使用零向量 {image_emb.shape}")
            
            # 🔧 第三步：读取文本嵌入
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
                    print(f"    {field}: 使用零向量 {field_emb.shape}")
            
            text_emb = np.concatenate(text_embeddings_list, axis=1)
            print(f"    文本嵌入合并: {text_emb.shape}")
            
            # 🔧 第四步：读取所有ID嵌入（按实际存在的特征）
            print(f"  🔧 读取ID嵌入...")
            id_embeddings_list = []
            
            # 🔥 按你提供的顺序读取所有ID特征
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
                'deep_features_21_embeddings',  # 注意：跳过20
                'deep_features_22_embeddings',
                'deep_features_23_embeddings',
                'deep_features_24_embeddings',
                'deep_features_25_embeddings',
                'deep_features_26_embeddings'
            ]
            
            print(f"    预期ID特征数: {len(id_feature_names)}")
            
            for feature_name in id_feature_names:
                if feature_name in f:
                    field_emb = f[feature_name][:num_samples]
                    id_embeddings_list.append(field_emb)
                    # 只打印前几个和关键的
                    if len(id_embeddings_list) <= 5 or 'deep_features_26' in feature_name:
                        print(f"    {feature_name}: {field_emb.shape}")
                else:
                    # 如果某个特征不存在，用零向量代替
                    field_emb = np.zeros((num_samples, 32), dtype=np.float32)
                    id_embeddings_list.append(field_emb)
                    print(f"    {feature_name}: 缺失，使用零向量 {field_emb.shape}")
            
            print(f"    实际读取ID特征数: {len(id_embeddings_list)}")
            
            # 合并所有ID嵌入
            id_emb = np.concatenate(id_embeddings_list, axis=1)
            print(f"    🎯 ID嵌入合并: {id_emb.shape}")  # 应该是 (num_samples, 28*32=896)
            
            # 🔧 第五步：读取标签和元数据
            labels = f['labels'][:num_samples]
            print(f"    标签: {labels.shape}")
            
            metadata = f['metadata'][:num_samples]
            print(f"    元数据: {metadata.shape}")
            
            # 处理元数据
            scenes = []
            user_ids = []
            item_ids = []
            
            print(f"    处理元数据...")
            for j in range(num_samples):
                meta = metadata[j]
                
                # 用户ID
                user_id_str = meta[0].decode('utf-8') if isinstance(meta[0], bytes) else str(meta[0])
                user_ids.append(user_id_str)
                
                # 物品ID  
                item_id_str = meta[1].decode('utf-8') if isinstance(meta[1], bytes) else str(meta[1])
                item_ids.append(item_id_str)
                
                # 场景ID
                scene_str = meta[2].decode('utf-8') if isinstance(meta[2], bytes) else str(meta[2])
                scene_id = int(scene_str) if scene_str.isdigit() else 0
                scenes.append(scene_id)
            
            # 准备返回数据
            single_file_data = {
                'image_embeddings': image_emb,
                'text_embeddings': text_emb,
                'id_embeddings': id_emb,
                'labels': labels,
                'scenes': np.array(scenes, dtype=np.int64),
                'user_ids': user_ids,
                'item_ids': item_ids
            }
            
            # 🔧 验证数据一致性
            print(f"  🔍 验证数据一致性:")
            for key, value in single_file_data.items():
                if key in ['user_ids', 'item_ids']:
                    print(f"    {key}: {len(value)} 个")
                else:
                    print(f"    {key}: {value.shape}")
            
            # 确保所有数组的第一维都一致
            shapes = []
            for key, value in single_file_data.items():
                if key in ['user_ids', 'item_ids']:
                    shapes.append(len(value))
                else:
                    shapes.append(value.shape[0])
            
            if len(set(shapes)) == 1:
                print(f"  ✅ 数据一致性验证通过: 所有特征都有 {shapes[0]:,} 样本")
            else:
                print(f"  ❌ 数据一致性验证失败: {set(shapes)}")
                raise ValueError("数据不一致！")
            
            total_positive = np.sum(labels)
            print(f"  📊 正样本: {int(total_positive):,} ({total_positive/num_samples:.2%})")
            
            return single_file_data
            
    except Exception as e:
        print(f"❌ 加载文件出错: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_memory_dataset_full_embeddings(data_dict):
    """
    从内存数据创建PyTorch数据集（全嵌入模式，支持LightGCN图专家）
    """
    print("🔄 创建PyTorch数据集...")
    
    # 转换为torch tensors
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
            print(f"  ⚠️ 缺少数据: {name}")
    
    # 🔥 新增：处理用户ID和物品ID
    user_ids = data_dict.get('user_ids', [])
    item_ids = data_dict.get('item_ids', [])
    
    # 创建自定义数据集类（支持字符串ID）
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
    print(f"✅ 数据集创建完成: {len(dataset):,} 样本（支持LightGCN图专家）")
    
    return dataset


def calculate_scene_auc(y_true, y_pred, scenes):
    """计算每个场景的AUC"""
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


# ============ Top-K防极化流式训练和评估函数 ============

def train_epoch_streaming_full_embeddings(model, train_files, optimizer, loss_function, device, 
                                         batch_size=1024, max_samples_per_file=None, shuffle_files=True):
    """
    🔥 Top-K防极化流式训练一个epoch（一次加载一个训练文件，全嵌入模式+LightGCN图专家）
    """
    model.train()
    total_loss = 0
    total_diversity_loss = 0  # 🔥 新增：多样性损失
    y_true_list = []
    y_pred_list = []
    scene_list = []
    total_batches = 0
    total_samples = 0
    
    # 打乱文件顺序（可选）
    files_to_process = train_files.copy()
    if shuffle_files:
        random.shuffle(files_to_process)
    
    print(f"🔄 Top-K防极化流式训练: 处理 {len(files_to_process)} 个文件")
    
    for file_idx, train_file in enumerate(files_to_process):
            print(f"\n📂 处理训练文件 [{file_idx+1}/{len(files_to_process)}]: {os.path.basename(train_file)}")
        
        # try:
            # 加载当前文件到内存
            single_file_data = load_single_h5_file_to_memory_full_embeddings(
                train_file, max_samples=max_samples_per_file
            )
            
            # 创建数据集和数据加载器
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
            print(f"  批次数: {file_batches}, 样本数: {file_samples:,}")
            
            # 训练当前文件的所有批次
            pbar = tqdm(train_loader, desc=f"训练文件{file_idx+1}")
            
            for batch in pbar:
                # 🔥 解包数据（包含用户ID和物品ID）
                image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch
                
                # 移到GPU
                image_emb = image_emb.to(device, non_blocking=True)
                text_emb = text_emb.to(device, non_blocking=True)
                id_emb = id_emb.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                scenes = scenes.to(device, non_blocking=True)
                
                # 🔥 Top-K防极化前向传播（返回预测和多样性损失）
                optimizer.zero_grad()
                
                # 检查模型是否支持Top-K
                # if hasattr(model, 'forward') and 'diversity_loss' in str(model.forward):
                    # Top-K防极化模型
                    # print("极化")
                predictions, diversity_loss = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
                    
                # else:
                #     print("非极化")
                #     # 普通模型
                #     predictions = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
                #     diversity_loss = torch.tensor(0.0, device=device)
                
                # 🔥 计算总损失（主损失 + 多样性损失）
                main_loss = loss_function(predictions, labels)
                
                # 获取多样性损失权重
                diversity_weight = getattr(model, 'diversity_weight', 0.01)
                total_loss_batch = main_loss + diversity_weight * diversity_loss
                
                # 反向传播
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 记录统计信息
                total_loss += main_loss.item()
                total_diversity_loss += diversity_loss.item()
                total_batches += 1
                
                # 收集所有预测结果
                with torch.no_grad():
                    y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
                    y_true = labels.detach().cpu().numpy()
                    scene_ids = scenes.detach().cpu().numpy()
                    
                    y_true_list.extend(y_true.tolist())
                    y_pred_list.extend(y_pred.tolist())
                    scene_list.extend(scene_ids.tolist())
                
                # 🔥 更新进度条显示（包含多样性损失）
                pbar.set_postfix({
                    'Loss': f'{main_loss.item():.4f}',
                    'Div': f'{diversity_loss.item():.5f}'
                })
            
            total_samples += file_samples
            
            # 立即清理当前文件的内存
            del single_file_data, train_dataset, train_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        # except Exception as e:
        #     print(f"  ❌ 处理文件 {train_file} 时出错: {e}")
        #     continue
    
    # 计算整体指标
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_diversity_loss = total_diversity_loss / total_batches if total_batches > 0 else 0
    
    overall_auc = 0
    scene_aucs = {}
    try:
        if len(set(y_true_list)) > 1 and len(y_true_list) > 0:
            overall_auc = roc_auc_score(y_true_list, y_pred_list)
            scene_aucs = calculate_scene_auc(y_true_list, y_pred_list, scene_list)
            print(f"  ✅ Top-K防极化训练AUC计算完成，基于 {len(y_true_list):,} 个样本")
    except Exception as e:
        print(f"⚠️ AUC计算错误: {e}")
    
    return avg_loss, overall_auc, scene_aucs, total_samples, avg_diversity_loss


def evaluate_epoch_streaming_full_embeddings(model, test_files, loss_function, device, 
                                            batch_size=1024, max_samples_per_file=None):
    """
    🔥 Top-K防极化流式评估一个epoch（逐文件加载测试数据，全嵌入模式+LightGCN图专家）
    """
    model.eval()
    total_loss = 0
    total_diversity_loss = 0  # 🔥 新增：多样性损失
    y_true_list = []
    y_pred_list = []
    scene_list = []
    total_batches = 0
    total_samples = 0
    
    print(f"🔄 Top-K防极化流式评估: 处理 {len(test_files)} 个测试文件")
    
    with torch.no_grad():
        for file_idx, test_file in enumerate(test_files):
            print(f"\n📂 处理测试文件 [{file_idx+1}/{len(test_files)}]: {os.path.basename(test_file)}")
            
            try:
                # 加载当前文件到内存
                single_file_data = load_single_h5_file_to_memory_full_embeddings(
                    test_file, max_samples=max_samples_per_file
                )
                
                # 创建数据集和数据加载器
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
                print(f"  批次数: {file_batches}, 样本数: {file_samples:,}")
                
                # 评估当前文件的所有批次
                pbar = tqdm(test_loader, desc=f"评估文件{file_idx+1}")
                
                for batch in pbar:
                    # 🔥 解包数据（包含用户ID和物品ID）
                    image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch
                    
                    # 移到GPU
                    image_emb = image_emb.to(device, non_blocking=True)
                    text_emb = text_emb.to(device, non_blocking=True)
                    id_emb = id_emb.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    scenes = scenes.to(device, non_blocking=True)
                    
                    # 🔥 Top-K防极化前向传播（返回预测和多样性损失）
                    # 检查模型是否支持Top-K
                    # if hasattr(model, 'forward') and 'diversity_loss' in str(model.forward):
                        # Top-K防极化模型
                    predictions, diversity_loss = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
                    # else:
                        # 普通模型
                        # predictions = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
                        # diversity_loss = torch.tensor(0.0, device=device)
                    
                    # 计算损失
                    main_loss = loss_function(predictions, labels)
                    total_loss += main_loss.item()
                    total_diversity_loss += diversity_loss.item()
                    total_batches += 1
                    
                    # 收集所有预测结果
                    y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
                    y_true = labels.detach().cpu().numpy()
                    scene_ids = scenes.detach().cpu().numpy()
                    
                    y_true_list.extend(y_true.tolist())
                    y_pred_list.extend(y_pred.tolist())
                    scene_list.extend(scene_ids.tolist())
                    
                    # 🔥 更新进度条显示（包含多样性损失）
                    pbar.set_postfix({
                        'Loss': f'{main_loss.item():.4f}',
                        'Div': f'{diversity_loss.item():.5f}'
                    })
                
                total_samples += file_samples
                
                # 立即清理当前文件的内存
                del single_file_data, test_dataset, test_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ❌ 处理文件 {test_file} 时出错: {e}")
                continue
    
    # 计算指标
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_diversity_loss = total_diversity_loss / total_batches if total_batches > 0 else 0
    
    overall_auc = 0
    scene_aucs = {}
    try:
        if len(set(y_true_list)) > 1 and len(y_true_list) > 0:
            overall_auc = roc_auc_score(y_true_list, y_pred_list)
            scene_aucs = calculate_scene_auc(y_true_list, y_pred_list, scene_list)
            print(f"  ✅ Top-K防极化测试AUC计算完成，基于 {len(y_true_list):,} 个样本")
    except Exception as e:
        print(f"⚠️ AUC计算错误: {e}")
    
    return avg_loss, overall_auc, scene_aucs, total_samples, avg_diversity_loss


# ============ 保持原有接口的兼容性函数 ============

def load_all_h5_data_to_memory_full_embeddings(h5_files, max_samples=None, use_mmap=True):
    """
    兼容原接口：实际使用流式加载（只加载第一个文件用于向后兼容）
    ⚠️ 这个函数现在改为流式处理，避免内存爆炸
    """
    print("🔄 兼容性接口：使用流式加载模式...")
    
    if not h5_files:
        raise ValueError("文件列表为空！")
    
    # 只加载第一个文件，避免内存爆炸
    first_file = h5_files[0]
    print(f"⚠️ 兼容模式：只加载第一个文件 {os.path.basename(first_file)}")
    
    return load_single_h5_file_to_memory_full_embeddings(first_file, max_samples)


def train_epoch_memory_full_embeddings(model, train_loader_or_files, optimizer, loss_function, device):
    """
    🔥 Top-K防极化兼容原接口：如果传入文件列表则使用流式训练，否则使用原逻辑
    """
    # 如果传入的是文件列表，转为流式训练
    if isinstance(train_loader_or_files, list):
        result = train_epoch_streaming_full_embeddings(
            model, train_loader_or_files, optimizer, loss_function, device
        )
        # 🔥 返回兼容格式（忽略多样性损失）
        if len(result) == 5:
            return result[:4]  # 去掉多样性损失
        return result
    
    # 如果传入的是DataLoader，保持原有逻辑（但支持LightGCN图专家+Top-K）
    print("⚠️ 使用DataLoader模式（支持LightGCN图专家+Top-K防极化）")
    train_loader = train_loader_or_files
    
    model.train()
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    scene_list = []
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        # 🔥 解包数据（支持新格式和旧格式）
        if len(batch) == 7:  # 新格式：包含user_id和item_id
            image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch
        else:  # 兼容旧格式
            image_emb, text_emb, id_emb, labels, scenes = batch[:5]
            user_ids, item_ids = None, None
        
        # 移到GPU
        image_emb = image_emb.to(device, non_blocking=True)
        text_emb = text_emb.to(device, non_blocking=True)
        id_emb = id_emb.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        scenes = scenes.to(device, non_blocking=True)
        
        # 🔥 Top-K防极化前向传播（传入用户ID和物品ID给LightGCN图专家）
        optimizer.zero_grad()
        
        # 检查模型是否支持Top-K
        if hasattr(model, 'forward') and 'diversity_loss' in str(model.forward):
            # Top-K防极化模型
            predictions, diversity_loss = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
            main_loss = loss_function(predictions, labels)
            diversity_weight = getattr(model, 'diversity_weight', 0.01)
            total_loss_batch = main_loss + diversity_weight * diversity_loss
        else:
            # 普通模型
            predictions = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
            total_loss_batch = loss_function(predictions, labels)
        
        # 反向传播
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 记录统计信息
        total_loss += total_loss_batch.item()
        num_batches += 1
        
        # 收集预测结果
        with torch.no_grad():
            y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            scene_ids = scenes.detach().cpu().numpy()
            
            y_true_list.extend(y_true.tolist())
            y_pred_list.extend(y_pred.tolist())
            scene_list.extend(scene_ids.tolist())
        
        pbar.set_postfix({'Loss': f'{total_loss_batch.item():.4f}'})
    
    # 计算指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    overall_auc = 0
    scene_aucs = {}
    try:
        if len(set(y_true_list)) > 1:
            overall_auc = roc_auc_score(y_true_list, y_pred_list)
            scene_aucs = calculate_scene_auc(y_true_list, y_pred_list, scene_list)
    except Exception as e:
        print(f"⚠️ AUC计算错误: {e}")
    
    return avg_loss, overall_auc, scene_aucs, len(y_true_list)


def evaluate_epoch_memory_full_embeddings(model, test_loader_or_files, loss_function, device):
    """
    🔥 Top-K防极化兼容原接口：如果传入文件列表则使用流式评估，否则使用原逻辑
    """
    # 如果传入的是文件列表，转为流式评估
    if isinstance(test_loader_or_files, list):
        result = evaluate_epoch_streaming_full_embeddings(
            model, test_loader_or_files, loss_function, device
        )
        # 🔥 返回兼容格式（忽略多样性损失）
        if len(result) == 5:
            return result[:4]  # 去掉多样性损失
        return result
    
    # 如果传入的是DataLoader，保持原有逻辑（但支持LightGCN图专家+Top-K）
    print("⚠️ 使用DataLoader模式（支持LightGCN图专家+Top-K防极化）")
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
            # 🔥 解包数据（支持新格式和旧格式）
            if len(batch) == 7:  # 新格式：包含user_id和item_id
                image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch
            else:  # 兼容旧格式
                image_emb, text_emb, id_emb, labels, scenes = batch[:5]
                user_ids, item_ids = None, None
            
            # 移到GPU
            image_emb = image_emb.to(device, non_blocking=True)
            text_emb = text_emb.to(device, non_blocking=True)
            id_emb = id_emb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            scenes = scenes.to(device, non_blocking=True)
            
            # 🔥 Top-K防极化前向传播（传入用户ID和物品ID给LightGCN图专家）
            # 检查模型是否支持Top-K
            if hasattr(model, 'forward') and 'diversity_loss' in str(model.forward):
                # Top-K防极化模型
                predictions, diversity_loss = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
            else:
                # 普通模型
                predictions = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
            
            # 计算损失
            loss = loss_function(predictions, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # 收集预测结果
            y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            scene_ids = scenes.detach().cpu().numpy()
            
            y_true_list.extend(y_true.tolist())
            y_pred_list.extend(y_pred.tolist())
            scene_list.extend(scene_ids.tolist())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # 计算指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    overall_auc = 0
    scene_aucs = {}
    try:
        if len(set(y_true_list)) > 1:
            overall_auc = roc_auc_score(y_true_list, y_pred_list)
            scene_aucs = calculate_scene_auc(y_true_list, y_pred_list, scene_list)
    except Exception as e:
        print(f"⚠️ AUC计算错误: {e}")
    
    return avg_loss, overall_auc, scene_aucs, len(y_true_list)


def print_memory_usage(stage=""):
    """打印内存使用情况"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"  📊 {stage} GPU内存: {gpu_memory:.2f}GB 已用 / {gpu_cached:.2f}GB 缓存")
    
    try:
        import psutil
        cpu_memory = psutil.virtual_memory()
        cpu_used_gb = cpu_memory.used / 1024**3
        cpu_total_gb = cpu_memory.total / 1024**3
        print(f"  📊 {stage} CPU内存: {cpu_used_gb:.2f}GB / {cpu_total_gb:.2f}GB ({cpu_memory.percent:.1f}%)")
    except ImportError:
        print(f"  📊 {stage} 无法获取CPU内存信息（需要安装psutil）")

def create_expert_analysis_dataloader(test_files, device, batch_size=1024, max_samples_per_file=None, max_files_for_analysis=2):
    """
    为专家权重分析创建数据加载器（从测试文件中采样）
    
    Args:
        test_files: 测试文件列表
        device: 设备
        batch_size: 批次大小
        max_samples_per_file: 每个文件最大样本数
        max_files_for_analysis: 用于分析的最大文件数
    
    Returns:
        data_loader: 用于专家分析的数据加载器
    """
    print(f"🔍 为专家权重分析准备数据加载器...")
    
    # 限制分析文件数量，避免过长时间
    analysis_files = test_files[:max_files_for_analysis]
    print(f"  使用 {len(analysis_files)} 个测试文件进行专家分析")
    
    # 只加载第一个文件用于分析（避免内存占用过多）
    if analysis_files:
        analysis_file = analysis_files[0]
        print(f"  📂 分析文件: {os.path.basename(analysis_file)}")
        
        try:
            # 加载数据
            single_file_data = load_single_h5_file_to_memory_full_embeddings(
                analysis_file, max_samples=max_samples_per_file
            )
            
            # 创建数据集
            analysis_dataset = create_memory_dataset_full_embeddings(single_file_data)
            
            # 🔥 修复：创建设备安全的数据加载器
            class DeviceSafeDataLoader:
                """设备安全的数据加载器包装器"""
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
                        
                        # 收集批次数据
                        for j in range(i, end_idx):
                            data = self.dataset[j]
                            # data = [image, text, id_emb, label, scene, user_id, item_id]
                            batch_data.append(data[:5])  # 前5个是张量
                            batch_user_ids.append(data[5])  # user_id
                            batch_item_ids.append(data[6])  # item_id
                        
                        # 堆叠张量并移到正确设备
                        if batch_data:
                            tensors = []
                            for tensor_idx in range(5):  # image, text, id_emb, label, scene
                                tensor_list = [item[tensor_idx] for item in batch_data]
                                stacked_tensor = torch.stack(tensor_list).to(self.device, non_blocking=True)
                                tensors.append(stacked_tensor)
                            
                            # 返回：image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids
                            yield (*tensors, batch_user_ids, batch_item_ids)
            
            # 创建设备安全的数据加载器
            analysis_loader = DeviceSafeDataLoader(analysis_dataset, batch_size, device)
            
            print(f"  ✅ 专家分析数据加载器创建完成: {len(analysis_dataset):,} 样本 (设备安全)")
            return analysis_loader
            
        except Exception as e:
            print(f"  ❌ 创建专家分析数据加载器失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None
# ============ 🔥 Top-K防极化主训练函数（改为流式模式）============

def train_model_memory_single_gpu_full_embeddings(model_class, model_kwargs, train_files, test_files,
                                                 epochs=50, lr=1e-4, batch_size=512, save_path="model/best_model.pt", 
                                                 early_stop=5, max_samples=None, use_mmap=True, 
                                                 enable_expert_analysis=True, expert_analysis_interval=1):
    """
    🔥 改为使用Top-K防极化流式训练的单GPU训练（全嵌入模式+LightGCN图专家+Top-K防极化+专家重要程度分析）
    注意：use_mmap参数保留但实际使用流式处理
    
    Args:
        model_class: 模型类
        model_kwargs: 模型参数字典
        train_files: 训练文件列表
        test_files: 测试文件列表
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批次大小
        save_path: 模型保存路径
        early_stop: 早停轮数
        max_samples: 每个文件最大样本数限制
        use_mmap: 兼容参数（实际使用流式处理）
        enable_expert_analysis: 是否启用专家重要程度分析
        expert_analysis_interval: 专家分析间隔（每几个epoch分析一次）
    """
    print(f"🚀 开始Top-K防极化流式单GPU训练（全嵌入模式+LightGCN图专家+Top-K防极化+专家重要程度分析，内存优化版本）...")
    print_memory_usage("初始")
    
    # 检查GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️ 使用GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("⚠️ 使用CPU训练")
    
    print(f"📦 Top-K防极化流式训练配置:")
    print(f"  训练方式: 逐文件流式处理（内存最优化）")
    print(f"  训练文件: {len(train_files)} 个")
    print(f"  测试文件: {len(test_files)} 个")
    print(f"  批次大小: {batch_size}")
    print(f"  每文件样本限制: {max_samples or '无限制'}")
    print(f"  🔥 支持LightGCN图专家（用户ID和物品ID）")
    print(f"  🎯 Top-K防极化: 三阶段动态训练策略")
    print(f"  🔥 专家重要程度分析: {'启用' if enable_expert_analysis else '禁用'}")
    if enable_expert_analysis:
        print(f"  📊 分析间隔: 每 {expert_analysis_interval} 个epoch")
    
    # 创建模型
    print("\n🏗️ 创建Top-K防极化模型...")
    model = model_class(**model_kwargs).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    if hasattr(model, 'use_lightgcn'):
        print(f"  🔥 LightGCN图专家: {'已启用' if model.use_lightgcn else '已禁用'}")
    if hasattr(model, 'topk_config'):
        print(f"  🎯 Top-K防极化: 已启用（动态K值策略）")
    
    # 🔥 新增：检查模型是否支持专家重要程度分析
    has_expert_analysis = hasattr(model, 'print_expert_weights_after_training')
    if enable_expert_analysis and has_expert_analysis:
        print(f"  🔥 专家重要程度分析: 已启用（支持Top-K防极化分析）")
    elif enable_expert_analysis and not has_expert_analysis:
        print(f"  ⚠️ 专家重要程度分析: 模型不支持，已禁用")
        enable_expert_analysis = False
    
    # 创建优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_function = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练设置
    best_auc = 0
    patience = 0
    train_history = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"\n🔥 开始Top-K防极化流式训练循环:")
    print(f"  设备: {device}")
    print(f"  学习率: {lr}")
    print(f"  训练轮数: {epochs}")
    print(f"  早停轮数: {early_stop}")
    print(f"  🔥 LightGCN图专家: {'启用' if model_kwargs.get('use_lightgcn', False) else '禁用'}")
    print(f"  🎯 Top-K防极化: {'启用' if hasattr(model, 'topk_config') else '未检测到'}")
    print(f"  🔥 专家重要程度分析: {'启用' if enable_expert_analysis else '未启用'}")
    print("=" * 60)
    
    # 训练循环
    for epoch in range(epochs):
        start_time = time.time()
        print_memory_usage(f"Epoch {epoch+1} 开始")
        
        # 🎯 更新Top-K防极化训练阶段
        if hasattr(model, 'update_epoch'):
            model.update_epoch(epoch)
            # 获取当前阶段信息
            if hasattr(model, 'get_training_phase_info'):
                phase_info = model.get_training_phase_info()
                print(f"\n🎯 Top-K训练阶段 [{epoch+1}]: {phase_info['training_phase']}")
                params = phase_info['current_topk_params']
                print(f"  参数: K={params['k']}, max_weight={params['max_weight']:.2f}, "
                      f"temp={params['temperature']:.1f}, α={params['alpha']:.2f}")
                print(f"  策略: {phase_info['phase_description']}")
        
        # 🔄 Top-K防极化流式训练（使用文件列表）
        train_result = train_epoch_streaming_full_embeddings(
            model, train_files, optimizer, loss_function, device, 
            batch_size=batch_size, max_samples_per_file=max_samples
        )
        
        # 🔥 处理返回结果（支持多样性损失）
        if len(train_result) == 5:
            train_loss, train_auc, train_scene_aucs, train_samples, train_diversity_loss = train_result
        else:
            train_loss, train_auc, train_scene_aucs, train_samples = train_result
            train_diversity_loss = 0.0
        
        print_memory_usage(f"Epoch {epoch+1} 训练后")
        
        # 🔄 Top-K防极化流式测试（使用文件列表）
        test_result = evaluate_epoch_streaming_full_embeddings(
            model, test_files, loss_function, device, 
            batch_size=batch_size, max_samples_per_file=max_samples
        )
        
        # 🔥 处理返回结果（支持多样性损失）
        if len(test_result) == 5:
            test_loss, test_auc, test_scene_aucs, test_samples, test_diversity_loss = test_result
        else:
            test_loss, test_auc, test_scene_aucs, test_samples = test_result
            test_diversity_loss = 0.0
        
        print_memory_usage(f"Epoch {epoch+1} 测试后")
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"\n📊 Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
        print(f"  训练 - Loss: {train_loss:.4f}, AUC: {train_auc:.4f} (样本: {train_samples:,})")
        print(f"  测试 - Loss: {test_loss:.4f}, AUC: {test_auc:.4f} (样本: {test_samples:,})")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.2e}")
        
        # 🔥 打印Top-K防极化相关信息
        if train_diversity_loss > 0 or test_diversity_loss > 0:
            print(f"  🎯 多样性损失 - 训练: {train_diversity_loss:.5f}, 测试: {test_diversity_loss:.5f}")
        
        # 打印场景AUC
        if train_scene_aucs:
            scene_auc_str = ", ".join([f"S{i}: {v:.3f}" if v else f"S{i}: N/A" 
                                     for i, v in enumerate([train_scene_aucs.get(f"scene_{j}_auc") for j in range(5)])])
            print(f"  训练场景AUC - {scene_auc_str}")
        
        if test_scene_aucs:
            scene_auc_str = ", ".join([f"S{i}: {v:.3f}" if v else f"S{i}: N/A" 
                                     for i, v in enumerate([test_scene_aucs.get(f"scene_{j}_auc") for j in range(5)])])
            print(f"  测试场景AUC - {scene_auc_str}")
        
        # 🔥 新增：专家重要程度分析
        if enable_expert_analysis and has_expert_analysis and (epoch + 1) % expert_analysis_interval == 0:
            print(f"\n🔥 进行专家重要程度分析...")
            try:
                # 创建专家分析数据加载器
                analysis_loader = create_expert_analysis_dataloader(
                    test_files, device, batch_size=batch_size, 
                    max_samples_per_file=max_samples, max_files_for_analysis=1
                )
                
                if analysis_loader is not None:
                    # 调用模型的专家重要程度分析功能
                    model.print_expert_weights_after_training(
                        data_loader=analysis_loader,
                        epoch=epoch + 1,
                        max_batches=3  # 限制分析批次数，避免过长时间
                    )
                    
                    # 清理分析数据加载器
                    del analysis_loader
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print(f"  ⚠️ 无法创建专家分析数据加载器，跳过本轮分析")
                    
            except Exception as e:
                print(f"  ❌ 专家重要程度分析失败: {e}")
                # 分析失败不影响训练继续
        
        # 记录历史
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'test_loss': test_loss,
            'test_auc': test_auc,
            'lr': scheduler.get_last_lr()[0],
            'train_samples': train_samples,
            'test_samples': test_samples,
            'train_diversity_loss': train_diversity_loss,  # 🔥 新增
            'test_diversity_loss': test_diversity_loss      # 🔥 新增
        }
        
        # 添加场景AUC
        for i in range(5):
            key = f"scene_{i}_auc"
            history_entry[f'train_{key}'] = train_scene_aucs.get(key) if train_scene_aucs else None
            history_entry[f'test_{key}'] = test_scene_aucs.get(key) if test_scene_aucs else None
        
        # 🎯 添加Top-K训练阶段信息
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
        
        # 保存最佳模型
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
                'feature_mode': 'topk_full_embeddings_streaming_with_lightgcn_expert_analysis',  # 🔥 更新标识
                'use_streaming': True,  # 🔥 标记使用流式处理
                'use_lightgcn': model_kwargs.get('use_lightgcn', False),  # 🔥 记录LightGCN使用情况
                'use_topk_antipolarization': hasattr(model, 'topk_config'),  # 🔥 记录Top-K使用情况
                'topk_config': getattr(model, 'topk_config', None) if hasattr(model, 'topk_config') else None,  # 🔥 保存Top-K配置
                'expert_analysis_enabled': enable_expert_analysis  # 🔥 记录专家分析状态
            }
            
            torch.save(checkpoint, save_path)
            print(f"  💾 保存最佳Top-K防极化模型 (AUC: {best_auc:.4f})")
            
        else:
            patience += 1
            print(f"  ⏳ 耐心等待: {patience}/{early_stop}")
        
        # 早停检查
        if patience >= early_stop:
            print(f"  🛑 早停触发!")
            break
        
        # 强制内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 🔥 新增：训练完成后进行最终的专家重要程度分析
    if enable_expert_analysis and has_expert_analysis:
        print(f"\n🔥 训练完成，进行最终专家重要程度分析...")
        try:
            # 创建专家分析数据加载器
            analysis_loader = create_expert_analysis_dataloader(
                test_files, device, batch_size=batch_size, 
                max_samples_per_file=max_samples, max_files_for_analysis=2
            )
            
            if analysis_loader is not None:
                # 调用模型的专家重要程度分析功能
                model.print_expert_weights_after_training(
                    data_loader=analysis_loader,
                    epoch="Final",  # 标记为最终分析
                    max_batches=5   # 最终分析使用更多批次
                )
                
                # 清理分析数据加载器
                del analysis_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"  ⚠️ 无法创建专家分析数据加载器，跳过最终分析")
                
        except Exception as e:
            print(f"  ❌ 最终专家重要程度分析失败: {e}")
    
    print(f"\n✅ Top-K防极化流式训练完成!")
    print(f"🏆 最佳测试AUC: {best_auc:.4f}")
    print(f"🔥 LightGCN图专家: {'已启用' if model_kwargs.get('use_lightgcn', False) else '未启用'}")
    print(f"🎯 Top-K防极化: {'已启用' if hasattr(model, 'topk_config') else '未检测到'}")
    print(f"🔥 专家重要程度分析: {'已启用' if enable_expert_analysis else '未启用'}")
    
    # 🎯 显示Top-K训练总结
    if hasattr(model, 'topk_config'):
        print(f"\n🎯 Top-K防极化训练总结:")
        config = model.topk_config
        print(f"  预热阶段(0-{config['warmup_epochs']}): K={config['warmup_k']}, 高探索性")
        print(f"  平衡阶段({config['warmup_epochs']+1}-{config['balance_epochs']}): K={config['balance_k']}, 动态收敛")
        print(f"  收敛阶段({config['balance_epochs']+1}+): K={config['final_k']}, 精细调优")
        print(f"  权重保护: 从{config['warmup_max_weight']}渐进到{config['final_max_weight']}")
        print(f"  🔥 专家重要程度: 每{expert_analysis_interval}个epoch分析一次，最终详细分析")
    
    # 保存训练历史
    history_path = save_path.replace('.pt', '_history.csv')
    pd.DataFrame(train_history).to_csv(history_path, index=False)
    print(f"📊 训练历史已保存到: {history_path}")
    print(f"📊 训练策略: Top-K防极化完全流式处理（逐文件加载+立即释放）")
    print(f"📊 特征模式: 全嵌入+LightGCN图专家+Top-K防极化+专家重要程度分析（图像256维 + 文本3840维 + ID928维 + 图嵌入）")
    print(f"💾 内存优势: 极致内存优化，支持无限大数据集")
    print(f"🔥 图专家架构: LightGCN按场景预训练图嵌入")
    print(f"🎯 防极化架构: 动态Top-K选择+多层防护机制")
    print(f"🔥 专家分析架构: 训练过程中实时监控专家重要程度")
    print_memory_usage("训练完成")
    
    return best_auc


# ============ 兼容性封装类（内存映射相关）============

class MemoryMappedDataset:
    """兼容性类：实际使用流式处理替代内存映射"""
    
    def __init__(self, mmap_data):
        print("⚠️ 兼容性模式：将内存映射数据转换为流式处理格式")
        
        # 从mmap数据提取基本信息
        self.length = len(mmap_data.get('labels', []))
        self.temp_dir = mmap_data.get('_temp_dir')
        
        # 将数据转换为标准格式（用于向后兼容）
        self.image_embeddings = mmap_data.get('image_embeddings')
        self.text_embeddings = mmap_data.get('text_embeddings')
        self.id_embeddings = mmap_data.get('id_embeddings')
        self.labels = mmap_data.get('labels')
        self.scenes = mmap_data.get('scenes')
        self.user_ids = mmap_data.get('user_ids', [])
        self.item_ids = mmap_data.get('item_ids', [])
        
        print(f"✅ 兼容性数据集: {self.length:,} 样本（支持LightGCN图专家+Top-K防极化）")
    
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
                # 如果是普通数组
                image = torch.tensor(self.image_embeddings[idx], dtype=torch.float32)
                text = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
                id_emb = torch.tensor(self.id_embeddings[idx], dtype=torch.float32)
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
                scene = torch.tensor(self.scenes[idx], dtype=torch.long)
                
            # 🔥 新增：用户ID和物品ID
            user_id = str(self.user_ids[idx]) if idx < len(self.user_ids) else "default_user"
            item_id = str(self.item_ids[idx]) if idx < len(self.item_ids) else "default_item"
            
            return image, text, id_emb, label, scene, user_id, item_id
        except Exception as e:
            print(f"⚠️ 读取样本 {idx} 时出错: {e}")
            # 返回默认样本
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
        """清理临时文件（兼容性方法）"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                print(f"🧹 兼容性临时文件已清理: {self.temp_dir}")
            except Exception as e:
                print(f"⚠️ 清理失败: {e}")
    
    def __del__(self):
        self.cleanup()