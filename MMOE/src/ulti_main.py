# train_main_heterogeneous.py - 异构专家流式训练主程序
import os
import torch
import torch.nn as nn
import numpy as np
import random
import warnings
import psutil
import gc
warnings.filterwarnings('ignore')

# 导入异构专家模型和训练模块
from multi_mmoe import HeterogeneousMMoE_FullEmbeddings, create_heterogeneous_mmoe_model
from multi_model_train import (
    train_model_memory_single_gpu_full_embeddings,
    load_single_h5_file_to_memory_full_embeddings,
    print_memory_usage
)
from utils import find_h5_files


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_memory_requirements_heterogeneous(train_files, test_files, max_samples=None):
    """检查异构专家模型的内存需求"""
    print(f"🔍 检查异构专家模型内存需求...")
    
    # 获取系统内存信息
    memory_info = psutil.virtual_memory()
    total_memory_gb = memory_info.total / (1024**3)
    available_memory_gb = memory_info.available / (1024**3)
    
    print(f"💾 系统内存:")
    print(f"  总内存: {total_memory_gb:.1f} GB")
    print(f"  可用内存: {available_memory_gb:.1f} GB")
    
    # 估算单文件内存需求
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
                
                # 异构专家模型特征内存估算
                sample_size_kb = 25  # 图像+文本+ID+用户ID+物品ID
                file_memory_gb = (file_samples * sample_size_kb) / (1024 * 1024)
                
                max_file_memory = max(max_file_memory, file_memory_gb)
                total_samples += file_samples
                
                print(f"  文件 {i+1}: {file_samples:,} 样本 -> ~{file_memory_gb:.2f} GB")
                
        except Exception as e:
            print(f"  ⚠️ 文件 {i+1} 检查失败: {e}")
    
    # 检查LightGCN图嵌入大小
    lightgcn_memory_gb = 0
    gcn_logs_dir = "/root/megrez-tmp/code/baseline+lightgcn/训练lightgcn/gcn_logs_incremental"
    if os.path.exists(gcn_logs_dir):
        print(f"🔍 检查LightGCN图嵌入内存需求...")
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
                    print(f"  场景{scene_id}: 用户嵌入{user_size:.3f}GB + 物品嵌入{item_size:.3f}GB")
            except:
                continue
        print(f"  LightGCN图嵌入总大小: {lightgcn_memory_gb:.3f} GB")
    else:
        print(f"⚠️ LightGCN目录不存在: {gcn_logs_dir}")
    
    # 异构专家模型内存需求计算
    pytorch_memory_gb = max_file_memory * 1.5
    model_memory_gb = 0.5  # 异构专家模型稍大
    training_overhead_gb = 2.0 + lightgcn_memory_gb  # 训练开销+图嵌入
    
    total_required_gb = pytorch_memory_gb + model_memory_gb + training_overhead_gb
    
    print(f"\n💾 异构专家模型内存需求分析:")
    print(f"  最大单文件内存: ~{max_file_memory:.2f} GB")
    print(f"  PyTorch处理开销: ~{pytorch_memory_gb:.2f} GB")
    print(f"  异构专家模型: ~{model_memory_gb:.2f} GB")
    print(f"  LightGCN图嵌入: ~{lightgcn_memory_gb:.3f} GB")
    print(f"  训练开销: ~{training_overhead_gb:.2f} GB")
    print(f"  峰值内存需求: ~{total_required_gb:.2f} GB")
    print(f"  🔥 异构优势: 专家特化，更高效的特征处理")
    
    # 检查是否有足够内存
    memory_ratio = total_required_gb / available_memory_gb
    
    if memory_ratio > 0.9:
        print(f"❌ 内存不足! 需要 {total_required_gb:.1f} GB，可用 {available_memory_gb:.1f} GB")
        return False, total_required_gb
    elif memory_ratio > 0.7:
        print(f"⚠️ 内存紧张 ({memory_ratio:.1%})")
        return True, total_required_gb
    else:
        print(f"✅ 内存充足 ({memory_ratio:.1%})")
        return True, total_required_gb


def optimize_memory_settings():
    """优化内存设置"""
    print("🔧 优化内存设置...")
    
    # 设置PyTorch内存管理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        print("✅ CUDA内存优化已启用")
    
    # 设置垃圾回收
    gc.set_threshold(700, 10, 10)
    print("✅ 垃圾回收优化已启用")


def check_environment():
    """检查训练环境（包含异构专家检查）"""
    print("🔍 检查异构专家训练环境...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA可用: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 8:
            print("⚠️ GPU内存较小，异构专家训练建议:")
            print("  - 减少batch_size到256-512")
            print("  - 启用mixed precision")
            print("  - 禁用部分专家")
        else:
            print("✅ GPU内存充足，适合异构专家训练")
    else:
        print("⚠️ CUDA不可用，将使用CPU（不推荐）")
    
    # 检查LightGCN图嵌入
    gcn_logs_dir = "/root/megrez-tmp/code/baseline+lightgcn/训练lightgcn/gcn_logs_incremental"
    if os.path.exists(gcn_logs_dir):
        scene_dirs = [d for d in os.listdir(gcn_logs_dir) if d.startswith('scene_')]
        print(f"🔥 LightGCN图专家检查:")
        print(f"  目录: {gcn_logs_dir}")
        print(f"  发现场景: {len(scene_dirs)} 个")
        
        available_scenes = 0
        for scene_dir in sorted(scene_dirs):
            try:
                scene_id = int(scene_dir.split('_')[1])
                scene_path = os.path.join(gcn_logs_dir, scene_dir)
                user_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_user_emb_final.npy")
                item_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_item_emb_final.npy")
                
                if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
                    print(f"    ✅ 场景{scene_id}: 图嵌入可用")
                    available_scenes += 1
                else:
                    print(f"    ⚠️ 场景{scene_id}: 图嵌入缺失")
            except:
                print(f"    ❌ 场景目录格式错误: {scene_dir}")
        
        if available_scenes > 0:
            print("✅ LightGCN图专家可用")
        else:
            print("⚠️ LightGCN图专家不可用")
    else:
        print(f"⚠️ LightGCN目录不存在: {gcn_logs_dir}")
    
    return True


def prepare_data_files(train_data_dir, test_data_dir, train_files=None, test_files=None):
    """准备数据文件"""
    print("🔍 准备数据文件...")
    
    # 查找训练和测试文件
    if train_files:
        train_files = [f for f in train_files if os.path.exists(f)]
        print(f"📂 使用指定的训练文件: {len(train_files)} 个")
    else:
        train_files = find_h5_files(train_data_dir)
        print(f"📂 从目录查找训练文件: {len(train_files)} 个")
    
    if test_files:
        test_files = [f for f in test_files if os.path.exists(f)]
        print(f"📂 使用指定的测试文件: {len(test_files)} 个")
    else:
        test_files = find_h5_files(test_data_dir)
        print(f"📂 从目录查找测试文件: {len(test_files)} 个")
    
    # 如果在同一个目录，需要区分训练和测试文件
    if train_data_dir == test_data_dir and not (train_files and test_files):
        all_files = find_h5_files(train_data_dir)
        train_files = [f for f in all_files if 'train' in os.path.basename(f).lower()]
        test_files = [f for f in all_files if 'test' in os.path.basename(f).lower()]
        
        if not train_files and not test_files:
            print("⚠️ 无法从文件名区分训练/测试文件，按比例分割")
            all_files.sort()
            mid_point = len(all_files) // 2
            train_files = all_files[:mid_point] if len(all_files) > 1 else all_files
            test_files = all_files[mid_point:] if len(all_files) > 1 else all_files
    
    if not train_files:
        print(f"❌ 未找到训练数据文件")
        return False, [], []
    
    if not test_files:
        print(f"❌ 未找到测试数据文件")
        return False, [], []
    
    print(f"\n📊 数据文件确认:")
    print(f"  训练文件: {len(train_files)} 个 (异构专家流式处理)")
    total_train_size = 0
    for i, f in enumerate(train_files):
        size_gb = os.path.getsize(f) / (1024**3)
        total_train_size += size_gb
        print(f"    [{i+1}] {os.path.basename(f)} ({size_gb:.2f} GB)")
    
    print(f"  测试文件: {len(test_files)} 个 (异构专家流式处理)")
    total_test_size = 0
    for i, f in enumerate(test_files):
        size_gb = os.path.getsize(f) / (1024**3)
        total_test_size += size_gb
        print(f"    [{i+1}] {os.path.basename(f)} ({size_gb:.2f} GB)")
    
    print(f"  文件总大小: {total_train_size + total_test_size:.2f} GB")
    print(f"  🔥 异构专家优势: 专业化处理，更高效的特征利用")
    
    return True, train_files, test_files


def create_heterogeneous_model_config(base_config):
    """创建异构专家模型配置"""
    print("🏗️ 配置异构专家多模态MMoE模型...")
    
    # 异构专家模型参数
    model_config = {
        'image_emb_dim': 256,      # 图像嵌入维度
        'text_emb_dim': 3840,      # 文本嵌入维度 (5*768)
        'id_emb_dim': 896,         # ID嵌入维度 (29*32)
        # 🔥 异构专家配置
        'use_lightgcn': True,      # 启用LightGCN图专家
        'gcn_logs_dir': "/root/megrez-tmp/code/baseline+lightgcn/训练lightgcn/gcn_logs_incremental",
        **base_config
    }
    
    print(f"🔍 异构专家配置:")
    print(f"  图像嵌入: {model_config['image_emb_dim']}维 -> 图像专家")
    print(f"  文本嵌入: {model_config['text_emb_dim']}维 -> 文本专家")
    print(f"  ID嵌入: {model_config['id_emb_dim']}维 -> ID专家")
    print(f"  总输入维度: {model_config['image_emb_dim'] + model_config['text_emb_dim'] + model_config['id_emb_dim']}维 -> 增强MLP专家")
    print(f"  🔥 LightGCN图专家: {'启用' if model_config['use_lightgcn'] else '禁用'}")
    if model_config['use_lightgcn']:
        print(f"  🔥 LightGCN目录: {model_config['gcn_logs_dir']}")
    
    return HeterogeneousMMoE_FullEmbeddings, model_config


def analyze_expert_performance(model, test_files, device, max_samples=5000):
    """分析异构专家性能"""
    print("\n📊 分析异构专家性能...")
    
    if not test_files:
        print("⚠️ 无测试文件，跳过专家分析")
        return
    
    model.eval()
    
    try:
        # 加载一个测试文件
        test_data = load_single_h5_file_to_memory_full_embeddings(
            test_files[0], max_samples=max_samples
        )
        
        # 准备数据
        image_emb = torch.tensor(test_data['image_embeddings'][:100], dtype=torch.float32).to(device)
        text_emb = torch.tensor(test_data['text_embeddings'][:100], dtype=torch.float32).to(device)
        id_emb = torch.tensor(test_data['id_embeddings'][:100], dtype=torch.float32).to(device)
        scenes = torch.tensor(test_data['scenes'][:100], dtype=torch.long).to(device)
        user_ids = test_data['user_ids'][:100]
        item_ids = test_data['item_ids'][:100]
        
        with torch.no_grad():
            # 获取专家权重分析
            expert_weights = model.get_expert_weights(
                image_emb, text_emb, id_emb, scenes, user_ids, item_ids
            )
            
            print("🎯 各场景专家权重分析:")
            for scene, weights in expert_weights.items():
                print(f"  {scene}:")
                sorted_experts = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                for expert, weight in sorted_experts:
                    print(f"    {expert}: {weight:.3f}")
            
            # 获取专家贡献分析
            contributions = model.get_expert_contributions(
                image_emb, text_emb, id_emb, scenes, user_ids, item_ids
            )
            
            print("\n🔍 专家贡献重要性分析:")
            for contrib_type, value in contributions.items():
                if not contrib_type.endswith('比例'):
                    print(f"  {contrib_type}: {value:.4f}")
            
            # 冷启动分析
            if '冷启动用户比例' in contributions:
                cold_ratio = contributions['冷启动用户比例']
                print(f"\n❄️ 冷启动分析:")
                print(f"  冷启动用户比例: {cold_ratio:.2%}")
                if 'LightGCN图专家' in contributions and '图专家_冷启动影响' in contributions:
                    normal_impact = contributions['LightGCN图专家']
                    cold_impact = contributions['图专家_冷启动影响']
                    print(f"  图专家正常影响: {normal_impact:.4f}")
                    print(f"  图专家冷启动影响: {cold_impact:.4f}")
                    print(f"  性能保持率: {(cold_impact/normal_impact)*100:.1f}%" if normal_impact > 0 else "  无法计算")
        
    except Exception as e:
        print(f"⚠️ 专家分析失败: {e}")


def main():
    """异构专家训练主函数"""
    print("🚀 开始异构专家单GPU多模态MMoE训练...")
    print("=" * 90)
    
    # 1. 设置随机种子
    set_seed(42)
    print("✅ 随机种子已设置")
    
    # 2. 优化内存设置
    optimize_memory_settings()
    
    # 3. 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        return
    
    # 4. 配置参数 - 异构专家配置
    config = {
        # 数据路径
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
        # 模型保存 (异构专家版本)
        'model_save_dir': "heterogeneous_model_full_embeddings",
        'model_save_name': "heterogeneous_mmoe_full_embeddings_best.pt",
        
        # 训练参数
        'epochs': 50,
        'learning_rate': 3e-5,  # 异构专家可能需要更小的学习率
        'batch_size': 4096,     # 适中的batch_size
        'early_stop': 8,
        'max_samples': None,
        
        'use_mmap': True,       # 兼容参数
        
        # 异构专家模型架构参数
        'model_config': {
            'expert_hidden_dim': 512,
            'tower_hidden_dims': [512, 256, 128],
            'output_dim': 1,
            'dropout': 0.3,
            'num_scenes': 5,
            # 🔥 异构专家配置
            'use_lightgcn': True,
            'gcn_logs_dir': "/root/megrez-tmp/code/baseline+lightgcn/训练lightgcn/gcn_logs_incremental"
        }
    }
    
    print("📋 异构专家训练配置:")
    for key, value in config.items():
        if key != 'model_config':
            print(f"  {key}: {value}")
    
    print("📋 异构专家架构配置:")
    for key, value in config['model_config'].items():
        marker = "🔥" if key in ['use_lightgcn', 'gcn_logs_dir'] else "  "
        print(f"  {marker} {key}: {value}")
    
    # 5. 准备数据文件
    success, train_files, test_files = prepare_data_files(
        config['train_data_dir'], 
        config['test_data_dir'],
        config.get('train_files'),
        config.get('test_files')
    )
    
    if not success:
        print("❌ 数据文件准备失败")
        return
    
    # 6. 检查内存需求
    print("\n" + "="*60)
    memory_ok, required_memory = check_memory_requirements_heterogeneous(
        train_files, test_files, config['max_samples']
    )
    
    if not memory_ok:
        print("❌ 资源不足，无法进行训练")
        print("💡 建议:")
        print("  1. 设置max_samples限制样本数量") 
        print("  2. 减小batch_size")
        print("  3. 禁用部分专家")
        print("  4. 增加系统内存")
        return
    
    # 7. 创建异构专家模型配置
    os.makedirs(config['model_save_dir'], exist_ok=True)
    save_path = os.path.join(config['model_save_dir'], config['model_save_name'])
    
    model_class, model_kwargs = create_heterogeneous_model_config(
        config['model_config']
    )
    
    print("\n" + "="*60)
    print("🔥 异构专家模式特点:")
    print("  ✨ 图像专家: CNN风格，专门处理视觉特征")
    print("  ✨ 文本专家: 注意力机制，专门处理语义特征")
    print("  ✨ ID专家: 因子分解，专门处理稀疏ID特征")
    print("  ✨ 图专家: LightGCN，专门处理协同过滤")
    print("  ✨ 增强MLP专家: Wide&Deep，处理特征交互")
    print("  🎯 门控机制智能选择最优专家组合")
    
    print("🔥 预期优势:")
    print("  📈 性能提升: 相比同构专家AUC提升2-5%")
    print("  🔍 可解释性: 专家级别的特征重要性分析")
    print("  🛡️ 鲁棒性: 冷启动时其他专家补偿")
    print("  🔧 灵活性: 可独立优化各专家超参数")
    
    print("="*60)
    
    try:
        # 8. 开始异构专家训练
        print(f"\n🚀 启动异构专家训练...")
        
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
        
        print(f"\n🎉 异构专家训练完成!")
        print("=" * 90)
        print(f"🏆 最佳AUC: {best_auc:.4f}")
        print(f"💾 模型已保存: {save_path}")
        
        # 9. 验证保存的模型
        if os.path.exists(save_path):
            model_size = os.path.getsize(save_path) / (1024**2)
            print(f"📊 模型文件大小: {model_size:.1f} MB")
            
            try:
                checkpoint = torch.load(save_path, map_location='cpu')
                print("✅ 模型验证通过")
                print(f"  训练轮数: {checkpoint['epoch']}")
                print(f"  最终AUC: {checkpoint['test_auc']:.4f}")
                print(f"  🔄 流式处理: {checkpoint.get('use_streaming', '未记录')}")
                print(f"  🔥 异构专家: {checkpoint.get('use_lightgcn', '未记录')}")
                
                # 10. 加载模型进行专家分析
                print(f"\n📊 加载模型进行专家性能分析...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # 创建模型并加载权重
                model = model_class(**model_kwargs).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # 分析专家性能
                analyze_expert_performance(model, test_files, device)
                
            except Exception as e:
                print(f"⚠️ 模型验证失败: {e}")
        
        # 11. 最终报告
        memory_info = psutil.virtual_memory()
        print(f"\n📊 最终系统状态:")
        print(f"  内存使用率: {memory_info.percent:.1f}%")
        print(f"  训练模式: 异构专家流式处理")
        
        print(f"\n📈 异构专家训练总结:")
        print(f"  🎯 最终AUC: {best_auc:.4f}")
        print(f"  🔥 架构创新: 5个异构专家各司其职")
        print(f"  ⚡ 训练效率: 流式处理，内存安全")
        print(f"  🎯 推荐场景: 需要高精度和可解释性的推荐系统")
        print(f"  🔍 分析能力: 专家级别的特征重要性分析")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 调试建议:")
        print(f"  - 检查数据文件是否完整")
        print(f"  - 减少max_samples参数")
        print(f"  - 减小batch_size")
        print(f"  - 检查各专家的参数设置")
        
        if config['model_config']['use_lightgcn']:
            print(f"  🔥 LightGCN相关:")
            print(f"    - 检查gcn_logs_incremental目录")
            print(f"    - 检查场景图嵌入文件")
            print(f"    - 尝试禁用LightGCN图专家")
        
        return
    
    print("\n✅ 异构专家多模态MMoE训练完成!")
    print(f"🚀 模型具备专家级特征处理能力!")
    print(f"🔥 异构架构让每种特征得到最适合的处理!")
    print(f"🎯 可用于高精度、高可解释性推荐服务!")


def quick_heterogeneous_test(data_files=None, max_samples=1000):
    """快速异构专家模型测试"""
    print(f"🧪 快速异构专家模型测试...")
    
    if data_files is None:
        data_files = [
            "/root/实验思路/baseline/embeddings/antm2c_10m_part0_train_embeddings.h5"
        ]
    
    try:
        # 加载测试数据
        data = load_single_h5_file_to_memory_full_embeddings(
            data_files[0], 
            max_samples=max_samples
        )
        
        print(f"✅ 数据加载成功!")
        print(f"  样本数: {len(data['labels']):,}")
        
        # 创建异构专家模型
        model = create_heterogeneous_mmoe_model()
        
        # 准备测试数据
        batch_size = min(10, len(data['labels']))
        image_emb = torch.tensor(data['image_embeddings'][:batch_size], dtype=torch.float32)
        text_emb = torch.tensor(data['text_embeddings'][:batch_size], dtype=torch.float32)
        id_emb = torch.tensor(data['id_embeddings'][:batch_size], dtype=torch.float32)
        scenes = torch.tensor(data['scenes'][:batch_size], dtype=torch.long)
        user_ids = data['user_ids'][:batch_size]
        item_ids = data['item_ids'][:batch_size]
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            output = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
            
        print(f"✅ 异构专家模型测试成功!")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min():.3f}, {output.max():.3f}]")
        
        # 测试专家分析
        expert_weights = model.get_expert_weights(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
        print(f"  专家权重分析: ✅")
        
        contributions = model.get_expert_contributions(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)
        print(f"  专家贡献分析: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "test":
            # 快速测试模式
            max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
            
            print(f"🧪 异构专家测试参数:")
            print(f"  样本数: {max_samples}")
            print(f"  🔥 测试异构专家架构")
            
            quick_heterogeneous_test(max_samples=max_samples)
            
        elif mode == "train":
            # 开始训练
            main()
            
        else:
            print("❌ 未知模式，支持的模式: test, train")
            print("💡 使用方法:")
            print("  python train_main_heterogeneous.py train                    # 开始异构专家训练")
            print("  python train_main_heterogeneous.py test [samples]           # 测试异构专家模型")
            print("    示例: python train_main_heterogeneous.py test 1000        # 测试1000样本")
            print("  🔥 特点: 5个异构专家各司其职，性能和可解释性双提升")
    else:
        # 默认运行训练
        print("🚀 启动异构专家单GPU训练...")
        main()