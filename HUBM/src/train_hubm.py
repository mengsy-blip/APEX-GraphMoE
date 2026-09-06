import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, ChineseCLIPModel, ChineseCLIPProcessor


REQUIRED_COLUMNS = [
    "user_id",
    "item_id",
    "item_entity_names",
    "item_title",
    "bill_entity_seq",
    "service_entity_seq",
    "query_entity_seq",
    "label",
]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_ddp(rank: int, world_size: int, config: Dict) -> None:
    os.environ.setdefault("MASTER_ADDR", config.get("master_addr", "localhost"))
    os.environ.setdefault("MASTER_PORT", str(config.get("master_port", "12355")))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"DDP initialized with {world_size} GPUs")


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_gpu_info() -> Tuple[bool, int]:
    if not torch.cuda.is_available():
        return False, 0
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPU(s)")
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")
    return gpu_count >= 2, gpu_count


def validate_training_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    df = pd.read_csv(data_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    return df


def build_item_csv(df: pd.DataFrame, output_dir: str, max_items: Optional[int] = None) -> str:
    if max_items:
        df = df.sample(n=min(max_items, len(df)), random_state=42)
    item_df = df[["item_id", "item_entity_names", "item_title"]].drop_duplicates()
    item_df = item_df.dropna(subset=["item_id"])
    os.makedirs(output_dir, exist_ok=True)
    item_csv_path = os.path.join(output_dir, "items.csv")
    item_df.to_csv(item_csv_path, index=False)
    print(f"Prepared {len(item_df)} unique items")
    return item_csv_path


class ChineseCLIPFineTuner(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.clip = ChineseCLIPModel.from_pretrained(model_name)

        for param in self.clip.vision_model.parameters():
            param.requires_grad = False
        for param in self.clip.text_model.parameters():
            param.requires_grad = False

        for layer in self.clip.vision_model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in self.clip.text_model.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, text_inputs: Dict[str, torch.Tensor], image_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.clip(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=image_inputs["pixel_values"],
            return_loss=True,
        )
        return outputs.loss


class CLIPDataset(Dataset):
    def __init__(self, csv_path: str, image_folder: str, processor: ChineseCLIPProcessor, max_text_length: int = 77):
        self.image_folder = image_folder
        self.processor = processor
        self.max_text_length = max_text_length
        self.data = []

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            item_id = str(row["item_id"])
            item_entities = "" if pd.isna(row.get("item_entity_names")) else str(row.get("item_entity_names"))
            item_title = "" if pd.isna(row.get("item_title")) else str(row.get("item_title"))
            text = f"{item_entities} {item_title}".strip()
            image_path = os.path.join(image_folder, f"{item_id}.png")
            if text and os.path.exists(image_path):
                self.data.append({"item_id": item_id, "text": text, "image_path": image_path})

        print(f"CLIP dataset size: {len(self.data)} valid item text-image pairs")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            inputs = self.processor(
                text=item["text"],
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
            )
            return {
                "text_inputs": {
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                },
                "image_inputs": {
                    "pixel_values": inputs["pixel_values"].squeeze(0),
                },
            }
        except Exception:
            return None


def collate_clip_batch(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return {
        "text_inputs": {
            "input_ids": torch.stack([item["text_inputs"]["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["text_inputs"]["attention_mask"] for item in batch]),
        },
        "image_inputs": {
            "pixel_values": torch.stack([item["image_inputs"]["pixel_values"] for item in batch]),
        },
    }


def train_clip_stage(rank: int, world_size: int, config: Dict) -> None:
    setup_ddp(rank, world_size, config)
    try:
        device = torch.device(f"cuda:{rank}")
        processor = ChineseCLIPProcessor.from_pretrained(config["clip_model"])
        dataset = CLIPDataset(config["item_csv"], config["image_folder"], processor)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset,
            batch_size=config["clip_batch_size"],
            sampler=sampler,
            collate_fn=collate_clip_batch,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
        )

        model = DDP(ChineseCLIPFineTuner(config["clip_model"]).to(device), device_ids=[rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate_clip", 1e-5))

        if rank == 0:
            print(f"Starting CLIP fine-tuning for {config['clip_epochs']} epochs")

        for epoch in range(config["clip_epochs"]):
            sampler.set_epoch(epoch)
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue
                text_inputs = {k: v.to(device) for k, v in batch["text_inputs"].items()}
                image_inputs = {k: v.to(device) for k, v in batch["image_inputs"].items()}
                optimizer.zero_grad()
                loss = model(text_inputs, image_inputs)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                if rank == 0 and batch_idx % 50 == 0:
                    print(f"Epoch {epoch + 1}, batch {batch_idx}, loss={loss.item():.4f}")

            if num_batches:
                avg_loss = total_loss / num_batches
                loss_tensor = torch.tensor(avg_loss, device=device)
                dist.all_reduce(loss_tensor)
                global_loss = loss_tensor.item() / world_size
                if rank == 0:
                    print(f"Epoch {epoch + 1} completed, average loss={global_loss:.4f}")

        if rank == 0:
            os.makedirs(config["output_dir"], exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(config["output_dir"], "clip_model.pt"))
            print("CLIP model saved")
    finally:
        cleanup_ddp()


def extract_item_embeddings(config: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    print("Extracting item embeddings")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ChineseCLIPFineTuner(config["clip_model"])
    model.load_state_dict(torch.load(os.path.join(config["output_dir"], "clip_model.pt"), map_location=device))
    model.eval().to(device)
    processor = ChineseCLIPProcessor.from_pretrained(config["clip_model"])

    df = pd.read_csv(config["item_csv"])
    if config.get("max_items"):
        df = df.head(config["max_items"])

    text_embeddings = {}
    image_embeddings = {}
    batch_size = 32
    for start_idx in tqdm(range(0, len(df), batch_size), desc="Embedding extraction"):
        batch_df = df.iloc[start_idx:start_idx + batch_size]
        batch_texts, batch_images, batch_ids = [], [], []

        for _, row in batch_df.iterrows():
            item_id = str(row["item_id"])
            item_entities = "" if pd.isna(row.get("item_entity_names")) else str(row.get("item_entity_names"))
            item_title = "" if pd.isna(row.get("item_title")) else str(row.get("item_title"))
            text = f"{item_entities} {item_title}".strip()
            image_path = os.path.join(config["image_folder"], f"{item_id}.png")
            if not text or not os.path.exists(image_path):
                continue
            try:
                batch_texts.append(text)
                batch_images.append(Image.open(image_path).convert("RGB"))
                batch_ids.append(item_id)
            except Exception:
                continue

        if not batch_texts:
            continue

        inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            text_embeds = model.clip.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_embeds = model.clip.get_image_features(pixel_values=inputs["pixel_values"])

        for i, item_id in enumerate(batch_ids):
            text_embeddings[item_id] = text_embeds[i].cpu().numpy()
            image_embeddings[item_id] = image_embeds[i].cpu().numpy()

    np.save(os.path.join(config["output_dir"], "text_embeddings.npy"), text_embeddings)
    np.save(os.path.join(config["output_dir"], "image_embeddings.npy"), image_embeddings)
    print(f"Extracted embeddings for {len(text_embeddings)} items")
    return text_embeddings, image_embeddings


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm1 = F.normalize(embeddings1, p=2, dim=1)
        norm2 = F.normalize(embeddings2, p=2, dim=1)
        sim = torch.mm(norm1, norm2.t()) / self.temperature

        if labels is None:
            target = torch.arange(embeddings1.size(0), device=embeddings1.device)
            return F.cross_entropy(sim, target)

        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        pos_mask = pos_mask - torch.eye(pos_mask.size(0), device=pos_mask.device)
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings1.device, requires_grad=True)

        exp_sim = torch.exp(sim)
        sum_exp = exp_sim.sum(dim=1, keepdim=True).squeeze()
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        pos_count = torch.clamp(pos_mask.sum(dim=1), min=1)
        return (-torch.log(pos_sim / (sum_exp * pos_count))).mean()


class HierarchicalUserBehaviorModel(nn.Module):
    def __init__(self, bert_model: str, text_dim: int, image_dim: int, hidden_size: int = 512):
        super().__init__()
        self.bill_bert = BertModel.from_pretrained(bert_model)
        self.service_bert = BertModel.from_pretrained(bert_model)
        self.query_bert = BertModel.from_pretrained(bert_model)

        self.user_fusion = nn.Sequential(
            nn.Linear(768 * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size),
        )
        self.text_projection = nn.Linear(text_dim, hidden_size)
        self.image_projection = nn.Linear(image_dim, hidden_size)
        self.modality_loss = ContrastiveLoss(0.07)
        self.semantic_loss = ContrastiveLoss(0.07)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def encode_user(self, bill_inputs: Dict[str, torch.Tensor], service_inputs: Dict[str, torch.Tensor], query_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        bill_embeds = self.bill_bert(**bill_inputs).last_hidden_state[:, 0]
        service_embeds = self.service_bert(**service_inputs).last_hidden_state[:, 0]
        query_embeds = self.query_bert(**query_inputs).last_hidden_state[:, 0]
        return self.user_fusion(torch.cat([bill_embeds, service_embeds, query_embeds], dim=-1))

    def encode_item(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        text_proj = self.text_projection(text_embeds)
        image_proj = self.image_projection(image_embeds)
        return (text_proj + image_proj) / 2

    def forward(self, bill_inputs, service_inputs, query_inputs, text_embeds, image_embeds, labels):
        user_embeds = self.encode_user(bill_inputs, service_inputs, query_inputs)
        text_proj = self.text_projection(text_embeds)
        image_proj = self.image_projection(image_embeds)
        item_embeds = (text_proj + image_proj) / 2

        modality_loss = self.modality_loss(text_proj, image_proj)
        semantic_loss = self.semantic_loss(user_embeds, item_embeds, labels)
        logits = self.classifier(torch.cat([user_embeds, item_embeds], dim=-1))
        classification_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float())
        total_loss = modality_loss + semantic_loss + classification_loss

        return {
            "total_loss": total_loss,
            "modality_loss": modality_loss,
            "semantic_loss": semantic_loss,
            "classification_loss": classification_loss,
            "logits": logits,
        }


class Stage2Dataset(Dataset):
    def __init__(self, data_path: str, text_embeddings: Dict[str, np.ndarray], image_embeddings: Dict[str, np.ndarray], tokenizer: BertTokenizer):
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.tokenizer = tokenizer
        self.data = []

        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = str(row["item_id"])
                if item_id in text_embeddings and item_id in image_embeddings:
                    self.data.append({
                        "bill_text": row.get("bill_entity_seq", ""),
                        "service_text": row.get("service_entity_seq", ""),
                        "query_text": row.get("query_entity_seq", ""),
                        "item_id": item_id,
                        "label": int(row["label"]),
                    })

        print(f"Stage-2 dataset size: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        bill_inputs = self.tokenizer(item["bill_text"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        service_inputs = self.tokenizer(item["service_text"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        query_inputs = self.tokenizer(item["query_text"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "bill_inputs": {k: v.squeeze(0) for k, v in bill_inputs.items()},
            "service_inputs": {k: v.squeeze(0) for k, v in service_inputs.items()},
            "query_inputs": {k: v.squeeze(0) for k, v in query_inputs.items()},
            "text_embedding": torch.tensor(self.text_embeddings[item["item_id"]], dtype=torch.float),
            "image_embedding": torch.tensor(self.image_embeddings[item["item_id"]], dtype=torch.float),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def train_stage2(rank: int, world_size: int, config: Dict) -> None:
    setup_ddp(rank, world_size, config)
    try:
        device = torch.device(f"cuda:{rank}")
        text_embeddings = np.load(os.path.join(config["output_dir"], "text_embeddings.npy"), allow_pickle=True).item()
        image_embeddings = np.load(os.path.join(config["output_dir"], "image_embeddings.npy"), allow_pickle=True).item()
        text_dim = next(iter(text_embeddings.values())).shape[0]
        image_dim = next(iter(image_embeddings.values())).shape[0]

        tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
        dataset = Stage2Dataset(config["data_path"], text_embeddings, image_embeddings, tokenizer)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset,
            batch_size=config["stage2_batch_size"],
            sampler=sampler,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
        )

        model = HierarchicalUserBehaviorModel(
            config["bert_model"],
            text_dim,
            image_dim,
            hidden_size=config.get("hidden_size", 512),
        ).to(device)
        model = DDP(model, device_ids=[rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate_stage2", 2e-5))

        best_loss = float("inf")
        if rank == 0:
            print(f"Starting HUBM stage-2 training for {config['stage2_epochs']} epochs")

        for epoch in range(config["stage2_epochs"]):
            sampler.set_epoch(epoch)
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                bill_inputs = {k: v.to(device) for k, v in batch["bill_inputs"].items()}
                service_inputs = {k: v.to(device) for k, v in batch["service_inputs"].items()}
                query_inputs = {k: v.to(device) for k, v in batch["query_inputs"].items()}
                text_embeds = batch["text_embedding"].to(device)
                image_embeds = batch["image_embedding"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(bill_inputs, service_inputs, query_inputs, text_embeds, image_embeds, labels)
                loss = outputs["total_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                if rank == 0 and batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch + 1}, batch {batch_idx}, "
                        f"total={loss.item():.4f}, "
                        f"modality={outputs['modality_loss'].item():.4f}, "
                        f"semantic={outputs['semantic_loss'].item():.4f}, "
                        f"classification={outputs['classification_loss'].item():.4f}"
                    )

            if not num_batches:
                continue
            avg_loss = total_loss / num_batches
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor)
            global_loss = loss_tensor.item() / world_size

            if rank == 0:
                print(f"Epoch {epoch + 1} completed, average loss={global_loss:.4f}")
                if global_loss < best_loss:
                    best_loss = global_loss
                    checkpoint_dir = os.path.join(config["output_dir"], "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": global_loss,
                            "epoch": epoch,
                        },
                        os.path.join(checkpoint_dir, "best_model.pt"),
                    )

        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(config["output_dir"], "final_model.pt"))
            print("HUBM stage-2 model saved")
    finally:
        cleanup_ddp()


def run_training(config: Dict) -> None:
    use_ddp, world_size = get_gpu_info()
    if not use_ddp:
        raise RuntimeError("DDP training requires at least two CUDA GPUs.")

    df = validate_training_data(config["data_path"])
    config["item_csv"] = build_item_csv(df, config["output_dir"], config.get("max_items"))

    clip_model_path = os.path.join(config["output_dir"], "clip_model.pt")
    if not os.path.exists(clip_model_path):
        mp.spawn(train_clip_stage, args=(world_size, config), nprocs=world_size, join=True)
    else:
        print("Existing CLIP model found; skipping CLIP fine-tuning")

    text_embedding_path = os.path.join(config["output_dir"], "text_embeddings.npy")
    image_embedding_path = os.path.join(config["output_dir"], "image_embeddings.npy")
    if not os.path.exists(text_embedding_path) or not os.path.exists(image_embedding_path):
        extract_item_embeddings(config)
    else:
        print("Existing item embeddings found; skipping extraction")

    mp.spawn(train_stage2, args=(world_size, config), nprocs=world_size, join=True)


def load_trained_model(model_path: str, config: Dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text_embeddings = np.load(os.path.join(config["output_dir"], "text_embeddings.npy"), allow_pickle=True).item()
    image_embeddings = np.load(os.path.join(config["output_dir"], "image_embeddings.npy"), allow_pickle=True).item()
    text_dim = next(iter(text_embeddings.values())).shape[0]
    image_dim = next(iter(image_embeddings.values())).shape[0]
    model = HierarchicalUserBehaviorModel(config["bert_model"], text_dim, image_dim, config.get("hidden_size", 512))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    return model, tokenizer, text_embeddings, image_embeddings, device


def evaluate_model(config: Dict, test_data_path: Optional[str] = None) -> Dict[str, float]:
    model_path = os.path.join(config["output_dir"], "checkpoints", "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(config["output_dir"], "final_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained HUBM model checkpoint was found.")

    model, tokenizer, text_embeddings, image_embeddings, device = load_trained_model(model_path, config)
    dataset = Stage2Dataset(test_data_path or config["data_path"], text_embeddings, image_embeddings, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=config.get("num_workers", 2))

    all_predictions, all_labels, all_probs = [], [], []
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            bill_inputs = {k: v.to(device) for k, v in batch["bill_inputs"].items()}
            service_inputs = {k: v.to(device) for k, v in batch["service_inputs"].items()}
            query_inputs = {k: v.to(device) for k, v in batch["query_inputs"].items()}
            text_embeds = batch["text_embedding"].to(device)
            image_embeds = batch["image_embedding"].to(device)
            labels = batch["label"].to(device)
            outputs = model(bill_inputs, service_inputs, query_inputs, text_embeds, image_embeds, labels)

            probs = torch.sigmoid(outputs["logits"]).squeeze()
            predictions = (probs > 0.5).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += outputs["total_loss"].item()

    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, zero_division=0),
        "recall": recall_score(all_labels, all_predictions, zero_division=0),
        "f1": f1_score(all_labels, all_predictions, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "avg_loss": total_loss / max(len(dataloader), 1),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def run_test(config: Dict) -> None:
    config = dict(config)
    config["output_dir"] = config.get("test_output_dir", "outputs/hubm_test")
    config["clip_epochs"] = 1
    config["stage2_epochs"] = 1
    config["clip_batch_size"] = min(config.get("clip_batch_size", 8), 8)
    config["stage2_batch_size"] = min(config.get("stage2_batch_size", 4), 4)
    config["max_items"] = config.get("test_max_items", 100)
    run_training(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate HUBM.")
    parser.add_argument("--config", default="configs/hubm_config.json")
    parser.add_argument("--mode", choices=["train", "test", "eval"], default="train")
    parser.add_argument("--test_data", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "train":
        run_training(config)
    elif args.mode == "test":
        run_test(config)
    elif args.mode == "eval":
        evaluate_model(config, args.test_data)


if __name__ == "__main__":
    main()
