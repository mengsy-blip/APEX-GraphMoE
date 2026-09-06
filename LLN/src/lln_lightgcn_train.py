import argparse
import gc
import json
import os
import pickle
import random
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, norm_adj):
        all_emb = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1).mean(dim=1)
        return torch.split(embs, [self.num_users, self.num_items])

    def predict(self, users, items, norm_adj):
        user_emb, item_emb = self.forward(norm_adj)
        return (user_emb[users] * item_emb[items]).sum(dim=1)


def load_bert_embeddings_for_scene(scene, bert_embeddings_dir):
    bert_file = os.path.join(bert_embeddings_dir, f"scene_{scene}_embeddings.pkl")
    if not os.path.exists(bert_file):
        print(f"BERT embedding file not found: {bert_file}")
        return None, None

    print(f"Loading BERT user-interest embeddings for scene {scene}")
    with open(bert_file, "rb") as f:
        bert_data = pickle.load(f)

    user_embeddings_dict = bert_data["user_embeddings_dict"]
    embedding_dim = bert_data["embedding_dim"]
    print(f"Loaded {len(user_embeddings_dict)} user embeddings with dim={embedding_dim}")
    return user_embeddings_dict, embedding_dim


def create_bert_embedding_matrix(user_embeddings_dict, user_id_to_idx, embedding_dim):
    num_users = len(user_id_to_idx)
    bert_embedding_matrix = np.zeros((num_users, embedding_dim), dtype=np.float32)

    print("Checking user-id matching between H5 data and BERT embeddings")
    h5_user_ids_sample = list(user_id_to_idx.keys())[:5]
    bert_user_ids_sample = list(user_embeddings_dict.keys())[:5]
    print(f"  H5 user-id examples: {h5_user_ids_sample}")
    print(f"  BERT user-id examples: {bert_user_ids_sample}")

    found_users = 0
    for user_id, user_idx in user_id_to_idx.items():
        matched = False

        if user_id in user_embeddings_dict:
            bert_embedding_matrix[user_idx] = user_embeddings_dict[user_id]
            matched = True
        elif str(user_id) in user_embeddings_dict:
            bert_embedding_matrix[user_idx] = user_embeddings_dict[str(user_id)]
            matched = True
        elif isinstance(user_id, str) and user_id.isdigit() and int(user_id) in user_embeddings_dict:
            bert_embedding_matrix[user_idx] = user_embeddings_dict[int(user_id)]
            matched = True
        elif isinstance(user_id, str):
            clean_id = user_id.lstrip("user_").lstrip("uid_").lstrip("u_")
            if clean_id in user_embeddings_dict:
                bert_embedding_matrix[user_idx] = user_embeddings_dict[clean_id]
                matched = True
            elif f"user_{user_id}" in user_embeddings_dict:
                bert_embedding_matrix[user_idx] = user_embeddings_dict[f"user_{user_id}"]
                matched = True

        if matched:
            found_users += 1
        else:
            bert_embedding_matrix[user_idx] = np.random.normal(0, 0.1, embedding_dim)

    match_rate = found_users / num_users * 100 if num_users else 0
    print(f"BERT embedding matrix built: {found_users}/{num_users} users matched ({match_rate:.1f}%)")
    if found_users == 0:
        h5_ids_set = set(str(uid) for uid in list(user_id_to_idx.keys())[:10])
        bert_ids_set = set(str(uid) for uid in list(user_embeddings_dict.keys())[:10])
        print(f"No user ids matched. First-10 string intersection: {h5_ids_set.intersection(bert_ids_set)}")
    elif found_users < num_users * 0.5:
        print("Warning: fewer than 50% of users matched BERT embeddings")

    return bert_embedding_matrix


def incremental_scene_graph_builder(h5_file_paths, similarity_threshold=0.5, top_k_similar=20):
    print("Building scene data incrementally from H5 metadata and labels")
    print(f"Input H5 files: {len(h5_file_paths)}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Top-K similar users: {top_k_similar}")

    scene_data = defaultdict(list)
    scene_user_ids = defaultdict(set)
    scene_item_ids = defaultdict(set)

    for file_idx, h5_file_path in enumerate(h5_file_paths):
        print(f"\nProcessing file [{file_idx + 1}/{len(h5_file_paths)}]: {os.path.basename(h5_file_path)}")
        if not os.path.exists(h5_file_path):
            print("  File not found; skipping")
            continue

        with h5py.File(h5_file_path, "r") as f:
            num_samples = len(f["metadata"])
            print(f"  Samples: {num_samples:,}")
            metadata = f["metadata"][:]
            labels = f["labels"][:]
            file_scene_stats = defaultdict(int)

            for sample_idx in range(num_samples):
                user_id = metadata[sample_idx][0].decode("utf-8") if isinstance(metadata[sample_idx][0], bytes) else str(metadata[sample_idx][0])
                item_id = metadata[sample_idx][1].decode("utf-8") if isinstance(metadata[sample_idx][1], bytes) else str(metadata[sample_idx][1])
                scene = metadata[sample_idx][2].decode("utf-8") if isinstance(metadata[sample_idx][2], bytes) else str(metadata[sample_idx][2])
                scene = scene.strip()
                if not scene:
                    continue

                sample_data = {
                    "user_id": user_id,
                    "item_id": item_id,
                    "scene": scene,
                    "label": int(labels[sample_idx]),
                    "file_source": os.path.basename(h5_file_path),
                }
                scene_data[scene].append(sample_data)
                scene_user_ids[scene].add(user_id)
                scene_item_ids[scene].add(item_id)
                file_scene_stats[scene] += 1

            print(f"  Scene distribution: {dict(file_scene_stats)}")
            filtered_samples = num_samples - sum(file_scene_stats.values())
            if filtered_samples > 0:
                print(f"  Filtered invalid-scene samples: {filtered_samples}")

    scene_final_stats = {}
    scene_mappings = {}

    print("\nScene summary")
    for scene in sorted(scene_data.keys()):
        samples = len(scene_data[scene])
        users = len(scene_user_ids[scene])
        items = len(scene_item_ids[scene])
        positive = sum(1 for sample in scene_data[scene] if sample["label"] == 1)
        scene_final_stats[scene] = {
            "samples": samples,
            "users": users,
            "items": items,
            "positive": positive,
            "negative": samples - positive,
        }
        print(f"  Scene {scene}: {samples:,} samples, {users:,} users, {items:,} items")

        scene_users = sorted(scene_user_ids[scene])
        scene_items = sorted(scene_item_ids[scene])
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(scene_users)}
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(scene_items)}
        scene_mappings[scene] = {
            "user_id_to_idx": user_id_to_idx,
            "item_id_to_idx": item_id_to_idx,
            "idx_to_user_id": {idx: user_id for user_id, idx in user_id_to_idx.items()},
            "idx_to_item_id": {idx: item_id for item_id, idx in item_id_to_idx.items()},
        }

    return scene_data, scene_mappings, scene_final_stats


def build_adj_matrix_with_bert_user_embeddings(
    df,
    num_users,
    num_items,
    bert_user_embeddings,
    user_id_to_idx,
    item_id_to_idx,
    top_k=20,
    use_positive_only=True,
    similarity_threshold=0.5,
    scene=None,
):
    print("Building a BERT-enhanced user-item graph")
    print(f"  BERT embedding shape: {bert_user_embeddings.shape}")
    print(f"  Similarity threshold: {similarity_threshold}")
    print(f"  Top-K similar users: {top_k}")
    print(f"  Scene: {scene}")

    max_users_per_item = 100 if scene == "2" else None
    df_filtered = df[df["label"] == 1].copy() if use_positive_only and "label" in df.columns else df.copy()

    user_indices = df_filtered["user_id"].map(user_id_to_idx).values
    item_indices = df_filtered["item_id"].map(item_id_to_idx).values
    valid_mask = (user_indices >= 0) & (item_indices >= 0)
    user_indices = user_indices[valid_mask]
    item_indices = item_indices[valid_mask] + num_users

    total_nodes = num_users + num_items
    if len(user_indices):
        if user_indices.max() >= num_users:
            raise ValueError(f"User index out of range: {user_indices.max()} >= {num_users}")
        if item_indices.max() >= total_nodes:
            raise ValueError(f"Item index out of range: {item_indices.max()} >= {total_nodes}")

    row_indices = [user_indices, item_indices]
    col_indices = [item_indices, user_indices]
    edge_weights = [
        np.ones(len(user_indices), dtype=np.float32),
        np.ones(len(item_indices), dtype=np.float32),
    ]

    item_user_map = defaultdict(set)
    for user_idx, item_idx in zip(user_indices, item_indices - num_users):
        item_user_map[item_idx].add(user_idx)

    new_row_indices = []
    new_col_indices = []
    new_edge_weights = []
    added_edges = 0
    sampled_items = 0

    for item_idx, users in tqdm(item_user_map.items(), desc="Adding semantic user-user edges"):
        users = list(users)
        if len(users) < 2:
            continue

        if max_users_per_item and len(users) > max_users_per_item:
            users = random.sample(users, max_users_per_item)
            sampled_items += 1

        valid_users = [u for u in users if u < len(bert_user_embeddings)]
        if len(valid_users) < 2:
            continue

        sim_matrix = cosine_similarity(bert_user_embeddings[valid_users])
        for i, u1 in enumerate(valid_users):
            sim_scores = [(j, score) for j, score in enumerate(sim_matrix[i]) if j != i]
            sim_scores.sort(key=lambda x: -x[1])

            added_count = 0
            for j_idx, score in sim_scores:
                if score > similarity_threshold and added_count < top_k:
                    u2 = valid_users[j_idx]
                    new_row_indices.extend([u1, u2])
                    new_col_indices.extend([u2, u1])
                    new_edge_weights.extend([float(score), float(score)])
                    added_edges += 2
                    added_count += 1

    if new_row_indices:
        row_indices.append(np.array(new_row_indices))
        col_indices.append(np.array(new_col_indices))
        edge_weights.append(np.array(new_edge_weights))

    row_indices = np.concatenate(row_indices)
    col_indices = np.concatenate(col_indices)
    edge_weights = np.concatenate(edge_weights)
    print(f"  Added semantic user-user edges: {added_edges:,}")
    if max_users_per_item:
        print(f"  Scene-2 sampled items: {sampled_items}/{len(item_user_map)}")

    adj = coo_matrix(
        (edge_weights, (row_indices, col_indices)),
        shape=(total_nodes, total_nodes),
        dtype=np.float32,
    )
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    norm_adj = adj.multiply(d_inv_sqrt[:, None]).multiply(d_inv_sqrt[None, :]).tocoo()

    indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
    values = torch.FloatTensor(norm_adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape)).coalesce()


def create_training_data_from_df(df, user_id_to_idx, item_id_to_idx, max_samples=None):
    df_valid = df[
        df["user_id"].isin(user_id_to_idx) & df["item_id"].isin(item_id_to_idx)
    ].copy()
    if max_samples and len(df_valid) > max_samples:
        df_valid = df_valid.sample(n=max_samples, random_state=42)

    train_users = df_valid["user_id"].map(user_id_to_idx).values
    train_items = df_valid["item_id"].map(item_id_to_idx).values
    train_labels = df_valid["label"].values.astype(np.float32)
    print(f"Training samples: {len(train_users):,}; positives: {train_labels.sum():,.0f}")
    return train_users, train_items, train_labels


def train_lightgcn_by_scene_incremental(
    h5_file_paths,
    save_dir,
    epochs=800,
    emb_dim=64,
    learning_rate=1e-3,
    early_stop_patience=15,
    max_train_samples=None,
    top_k_similar=20,
    device_id=0,
    similarity_threshold=0.65,
    bert_embeddings_dir=None,
):
    if bert_embeddings_dir is None:
        raise ValueError("bert_embeddings_dir is required")

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scene_data, scene_mappings, scene_stats = incremental_scene_graph_builder(
        h5_file_paths,
        similarity_threshold=similarity_threshold,
        top_k_similar=top_k_similar,
    )
    os.makedirs(save_dir, exist_ok=True)

    scene_results = {}
    for scene_idx, scene in enumerate(sorted(scene_data.keys())):
        print(f"\nTraining scene {scene} [{scene_idx + 1}/{len(scene_data)}]")
        mappings = scene_mappings[scene]
        stats = scene_stats[scene]

        bert_dict, bert_embedding_dim = load_bert_embeddings_for_scene(scene, bert_embeddings_dir)
        if bert_dict is None:
            print(f"Skipping scene {scene}: no BERT embeddings")
            continue

        bert_matrix = create_bert_embedding_matrix(
            bert_dict,
            mappings["user_id_to_idx"],
            bert_embedding_dim,
        )
        scene_df = pd.DataFrame(scene_data[scene])
        scene_save_dir = os.path.join(save_dir, f"scene_{scene}")
        os.makedirs(scene_save_dir, exist_ok=True)

        for mapping_name, mapping_data in mappings.items():
            with open(os.path.join(scene_save_dir, f"{mapping_name}.pkl"), "wb") as f:
                pickle.dump(mapping_data, f)

        norm_adj = build_adj_matrix_with_bert_user_embeddings(
            scene_df,
            stats["users"],
            stats["items"],
            bert_matrix,
            mappings["user_id_to_idx"],
            mappings["item_id_to_idx"],
            top_k=top_k_similar,
            similarity_threshold=similarity_threshold,
            scene=scene,
        ).to(device)

        train_users, train_items, train_labels = create_training_data_from_df(
            scene_df,
            mappings["user_id_to_idx"],
            mappings["item_id_to_idx"],
            max_train_samples,
        )
        train_users = torch.LongTensor(train_users).to(device)
        train_items = torch.LongTensor(train_items).to(device)
        train_labels = torch.FloatTensor(train_labels).to(device)

        model = LightGCN(stats["users"], stats["items"], emb_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = float("inf")
        patience = 0

        for epoch in tqdm(range(epochs), desc=f"Scene {scene}"):
            model.train()
            preds = model.predict(train_users, train_items, norm_adj)
            loss = F.binary_cross_entropy_with_logits(preds, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:3d}, loss={current_loss:.6f}")

            if current_loss < best_loss:
                best_loss = current_loss
                patience = 0
                user_emb, item_emb = model.forward(norm_adj)
                np.save(os.path.join(scene_save_dir, f"lightgcn_scene{scene}_user_emb_best.npy"), user_emb.detach().cpu().numpy())
                np.save(os.path.join(scene_save_dir, f"lightgcn_scene{scene}_item_emb_best.npy"), item_emb.detach().cpu().numpy())
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if torch.cuda.is_available() and (epoch + 1) % 20 == 0:
                torch.cuda.empty_cache()

        user_emb, item_emb = model.forward(norm_adj)
        final_user_emb = user_emb.detach().cpu().numpy()
        final_item_emb = item_emb.detach().cpu().numpy()
        np.save(os.path.join(scene_save_dir, f"lightgcn_scene{scene}_user_emb_final.npy"), final_user_emb)
        np.save(os.path.join(scene_save_dir, f"lightgcn_scene{scene}_item_emb_final.npy"), final_item_emb)
        np.save(os.path.join(scene_save_dir, "bert_user_embeddings.npy"), bert_matrix)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "num_users": stats["users"],
                "num_items": stats["items"],
                "emb_dim": emb_dim,
                "scene": scene,
                "similarity_threshold": similarity_threshold,
                "bert_embedding_dim": bert_embedding_dim,
                "top_k_similar": top_k_similar,
            },
            os.path.join(scene_save_dir, f"lightgcn_scene{scene}_model_final.pt"),
        )

        scene_result = {
            "scene": scene,
            "num_samples": stats["samples"],
            "num_users": stats["users"],
            "num_items": stats["items"],
            "positive_samples": stats["positive"],
            "negative_samples": stats["negative"],
            "best_loss": best_loss,
            "final_epoch": epoch + 1,
            "user_emb_shape": final_user_emb.shape,
            "item_emb_shape": final_item_emb.shape,
            "similarity_threshold": similarity_threshold,
            "bert_embedding_dim": bert_embedding_dim,
            "top_k_similar": top_k_similar,
            "processing_method": "incremental_with_bert_embeddings",
        }
        scene_results[scene] = scene_result
        with open(os.path.join(scene_save_dir, "scene_stats.json"), "w", encoding="utf-8") as f:
            json.dump(scene_result, f, indent=2, default=str)

        del model, optimizer, norm_adj, train_users, train_items, train_labels, bert_matrix
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_stats = {
        "total_scenes": len(scene_data),
        "similarity_threshold": similarity_threshold,
        "top_k_similar": top_k_similar,
        "emb_dim": emb_dim,
        "processing_method": "incremental_with_bert_user_interest_embeddings",
        "bert_embeddings_dir": bert_embeddings_dir,
        "scenes": scene_results,
        "device": str(device),
        "training_config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "early_stop_patience": early_stop_patience,
            "max_train_samples": max_train_samples,
        },
    }
    with open(os.path.join(save_dir, "global_training_stats.json"), "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2, default=str)

    return scene_results


def get_h5_files_from_directory(directory_path):
    if os.path.isfile(directory_path):
        return [directory_path] if directory_path.endswith(".h5") else []
    if os.path.isdir(directory_path):
        return sorted(
            os.path.join(directory_path, file)
            for file in os.listdir(directory_path)
            if file.endswith(".h5")
        )
    return []


def load_scene_model_and_embeddings(scene, save_dir):
    scene_dir = os.path.join(save_dir, f"scene_{scene}")
    if not os.path.exists(scene_dir):
        raise ValueError(f"Scene model directory does not exist: {scene_dir}")

    user_emb = np.load(os.path.join(scene_dir, f"lightgcn_scene{scene}_user_emb_final.npy"))
    item_emb = np.load(os.path.join(scene_dir, f"lightgcn_scene{scene}_item_emb_final.npy"))
    bert_path = os.path.join(scene_dir, "bert_user_embeddings.npy")
    bert_emb = np.load(bert_path) if os.path.exists(bert_path) else None
    model_state = torch.load(os.path.join(scene_dir, f"lightgcn_scene{scene}_model_final.pt"))

    mappings = {}
    for mapping_file in ["user_id_to_idx.pkl", "item_id_to_idx.pkl", "idx_to_user_id.pkl", "idx_to_item_id.pkl"]:
        mapping_path = os.path.join(scene_dir, mapping_file)
        if os.path.exists(mapping_path):
            with open(mapping_path, "rb") as f:
                mappings[mapping_file.replace(".pkl", "")] = pickle.load(f)

    stats_path = os.path.join(scene_dir, "scene_stats.json")
    stats = None
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    return user_emb, item_emb, bert_emb, model_state, mappings, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLN with BERT-enhanced LightGCN.")
    parser.add_argument("--h5_files", nargs="+", required=True)
    parser.add_argument("--bert_embeddings_dir", required=True)
    parser.add_argument("--save_dir", default="LLN/outputs/gcn_logs_incremental_bert")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--top_k_similar", type=int, default=20)
    parser.add_argument("--similarity_threshold", type=float, default=0.65)
    return parser.parse_args()


def main():
    args = parse_args()
    h5_file_paths = []
    for path in args.h5_files:
        h5_file_paths.extend(get_h5_files_from_directory(path))

    if not h5_file_paths:
        raise FileNotFoundError("No valid H5 files were found.")
    if not os.path.isdir(args.bert_embeddings_dir):
        raise FileNotFoundError(f"BERT embedding directory not found: {args.bert_embeddings_dir}")

    train_lightgcn_by_scene_incremental(
        h5_file_paths=h5_file_paths,
        save_dir=args.save_dir,
        epochs=args.epochs,
        emb_dim=args.emb_dim,
        learning_rate=args.learning_rate,
        early_stop_patience=args.early_stop_patience,
        max_train_samples=args.max_train_samples,
        top_k_similar=args.top_k_similar,
        device_id=args.device_id,
        similarity_threshold=args.similarity_threshold,
        bert_embeddings_dir=args.bert_embeddings_dir,
    )


if __name__ == "__main__":
    main()
