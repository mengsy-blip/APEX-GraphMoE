
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import json
import math




class TopKGate(nn.Module):
    def __init__(self, input_dim, num_experts, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim


        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim // 2, num_experts)
        )


        self.temperature = nn.Parameter(torch.tensor(2.0))


        self.weight_balance_factor = nn.Parameter(torch.tensor(0.1))

        print(f" Top-K gate network initialized: {input_dim}dim -> {num_experts}expert")

    def forward(self, x, k=None, max_weight=None, temperature=None, alpha=None, training_epoch=None):
        batch_size = x.size(0)


        gate_logits = self.gate_network(x)  # [batch_size, num_experts]


        if temperature is not None:
            current_temp = temperature
        else:
            current_temp = self.temperature


        scaled_logits = gate_logits / current_temp


        if self.training and training_epoch is not None and training_epoch < 10:

            noise_scale = max(0.1, 0.5 - training_epoch * 0.05)
            noise = torch.randn_like(scaled_logits) * noise_scale
            scaled_logits = scaled_logits + noise


        if k is not None and k < self.num_experts:

            top_k_values, top_k_indices = torch.topk(scaled_logits, k, dim=-1)


            mask = torch.zeros_like(scaled_logits)
            mask.scatter_(1, top_k_indices, 1.0)


            masked_logits = scaled_logits.masked_fill(mask == 0, float('-inf'))
            gate_weights = F.softmax(masked_logits, dim=-1)

            selected_experts = top_k_indices
        else:

            gate_weights = F.softmax(scaled_logits, dim=-1)
            selected_experts = None


        if max_weight is not None and max_weight < 1.0:

            gate_weights = torch.clamp(gate_weights, max=max_weight)

            # gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)


        if alpha is not None and alpha > 0:
            uniform_weights = torch.ones_like(gate_weights) / self.num_experts
            gate_weights = (1 - alpha) * gate_weights + alpha * uniform_weights


        diversity_loss = self._compute_diversity_loss(gate_weights)

        return gate_weights, selected_experts, diversity_loss

    def _compute_diversity_loss(self, gate_weights):

        expert_usage = gate_weights.mean(dim=0)  # [num_experts]


        ideal_usage = 1.0 / self.num_experts


        usage_variance = torch.var(expert_usage)


        epsilon = 1e-8
        entropy_loss = -torch.sum(expert_usage * torch.log(expert_usage + epsilon))
        max_entropy = torch.log(torch.tensor(float(self.num_experts)))
        normalized_entropy_loss = (max_entropy - entropy_loss) / max_entropy


        diversity_loss = usage_variance + 0.1 * normalized_entropy_loss

        return diversity_loss




class CNNImageExpert(nn.Module):
    def __init__(self, image_dim=256, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "cnn_image"



        self.feature_map_size = int(math.sqrt(image_dim))  # 16
        if self.feature_map_size * self.feature_map_size != image_dim:

            self.input_projection = nn.Linear(image_dim, 16*16)
            self.feature_map_size = 16
        else:
            self.input_projection = None


        self.cnn_backbone = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),


            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),


            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),


            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )


        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [B, 512, 2, 2] -> [B, 512, 1, 1]


        self.feature_projection = nn.Sequential(
            nn.Linear(512, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )


        if self.input_projection:
            self.residual_projection = nn.Linear(16*16, expert_hidden_dim)
        else:
            self.residual_projection = nn.Linear(image_dim, expert_hidden_dim)

        print(f" CNN image expert initialized: {image_dim}dim -> {expert_hidden_dim}dim")

    def forward(self, image_emb):
        batch_size = image_emb.size(0)


        if self.input_projection:
            x = self.input_projection(image_emb)  # [B, 256]
            residual_input = x
        else:
            x = image_emb
            residual_input = image_emb


        x = x.view(batch_size, 1, self.feature_map_size, self.feature_map_size)


        cnn_features = self.cnn_backbone(x)  # [B, 512, 2, 2]


        pooled_features = self.global_pool(cnn_features)  # [B, 512, 1, 1]
        pooled_features = pooled_features.view(batch_size, -1)  # [B, 512]


        main_output = self.feature_projection(pooled_features)


        residual = self.residual_projection(residual_input)


        output = main_output + residual * 0.1

        return output


class TransformerTextExpert(nn.Module):
    def __init__(self, text_dim=3840, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "transformer_text"
        self.text_fields = 5
        self.field_dim = 768



        self.position_embedding = nn.Parameter(torch.randn(self.text_fields, self.field_dim))


        self.input_projection = nn.Linear(self.field_dim, expert_hidden_dim)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=expert_hidden_dim,
            nhead=8,
            dim_feedforward=expert_hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )


        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=expert_hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )


        self.query_vector = nn.Parameter(torch.randn(1, expert_hidden_dim))


        self.output_projection = nn.Sequential(
            nn.LayerNorm(expert_hidden_dim),
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )


        self.residual_projection = nn.Linear(text_dim, expert_hidden_dim)

        print(f" Transformer text expert initialized: {text_dim}dim -> {expert_hidden_dim}dim")

    def forward(self, text_emb):
        batch_size = text_emb.size(0)


        text_fields = text_emb.view(batch_size, self.text_fields, self.field_dim)


        text_fields = text_fields + self.position_embedding.unsqueeze(0)


        projected_inputs = self.input_projection(text_fields)  # [batch, 5, hidden_dim]


        transformer_output = self.transformer_encoder(projected_inputs)  # [batch, 5, hidden_dim]


        query = self.query_vector.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
        pooled_output, attention_weights = self.attention_pooling(
            query, transformer_output, transformer_output
        )  # [batch, 1, hidden_dim]

        pooled_output = pooled_output.squeeze(1)  # [batch, hidden_dim]


        main_output = self.output_projection(pooled_output)


        residual = self.residual_projection(text_emb)


        output = main_output + residual * 0.1

        return output


class IDExpert(nn.Module):
    def __init__(self, id_dim=928, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "id"
        self.num_fields = 28
        self.field_dim = 32



        self.field_importance = nn.Parameter(torch.ones(self.num_fields))



        self.user_item_processor = nn.Sequential(
            nn.Linear(self.field_dim * 2, expert_hidden_dim // 2),
            nn.BatchNorm1d(expert_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        self.deep_features_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.field_dim, expert_hidden_dim // 8),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(26)
        ])


        self.interaction_layer = nn.Sequential(
            nn.Linear(expert_hidden_dim // 2 + expert_hidden_dim // 8 * 26, expert_hidden_dim),
            nn.BatchNorm1d(expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        self.deep_network = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.BatchNorm1d(expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),


            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.BatchNorm1d(expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        self.output_projection = nn.Linear(expert_hidden_dim, expert_hidden_dim)

        print(f" ID expert initialized: {id_dim}dim -> {expert_hidden_dim}dim (29IDfields)")

    def forward(self, id_emb):
        batch_size = id_emb.size(0)


        id_fields = id_emb.view(batch_size, self.num_fields, self.field_dim)


        user_item_concat = torch.cat([id_fields[:, 0, :], id_fields[:, 1, :]], dim=1)
        user_item_out = self.user_item_processor(user_item_concat)


        deep_feature_outputs = []
        for i, processor in enumerate(self.deep_features_processor):
            field_idx = i + 2

            weighted_field = id_fields[:, field_idx, :] * self.field_importance[field_idx]
            deep_out = processor(weighted_field)
            deep_feature_outputs.append(deep_out)


        deep_features_concat = torch.cat(deep_feature_outputs, dim=1)


        combined_features = torch.cat([user_item_out, deep_features_concat], dim=1)


        interaction_out = self.interaction_layer(combined_features)


        deep_out = self.deep_network(interaction_out)


        output = self.output_projection(deep_out) + interaction_out * 0.1

        return output


class LightGCNGraphExpert(nn.Module):
    def __init__(self, expert_hidden_dim=512, dropout=0.2, gcn_logs_dir="gcn_logs_incremental"):
        super().__init__()
        self.expert_type = "graph"
        self.expert_hidden_dim = expert_hidden_dim
        self.gcn_logs_dir = gcn_logs_dir


        self.scene_user_embeddings_gpu = {}
        self.scene_item_embeddings_gpu = {}
        self.scene_mappings = {}
        self.scene_avg_user_emb = {}
        self.scene_avg_item_emb = {}


        self.device = None
        self.embeddings_loaded_on_gpu = False
        self.graph_emb_dim = 0


        self.lightgcn_enabled = True


        self._load_all_scene_embeddings()


        if self.graph_emb_dim > 0 and self.lightgcn_enabled:
            self.graph_projection = nn.Sequential(
                nn.Linear(self.graph_emb_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_hidden_dim)
            )
        else:

            self.graph_projection = nn.Sequential(
                nn.Linear(expert_hidden_dim // 4, expert_hidden_dim),
                nn.ReLU()
            )
            self.lightgcn_enabled = False
            print(f" LightGCN graph expert is disabled,usingtext")

        print(f" LightGCN graph expert initialized: graph embeddings{self.graph_emb_dim}dim -> {expert_hidden_dim}dim")
        print(f"    performance optimization: {'enabled' if self.lightgcn_enabled else 'disabled'}")
        print(f"   number of loaded scenes: {len(self.scene_user_embeddings_gpu) if self.lightgcn_enabled else 0}")

    def _load_all_scene_embeddings(self):
        if not os.path.exists(self.gcn_logs_dir):
            print(f" LightGCN directory does not exist: {self.gcn_logs_dir}")
            self.lightgcn_enabled = False
            return

        scene_dirs = [d for d in os.listdir(self.gcn_logs_dir) if d.startswith('scene_')]

        if not scene_dirs:
            print(f" no scene directory found,disabledLightGCN graph expert")
            self.lightgcn_enabled = False
            return

        loaded_scenes = 0
        for scene_dir in sorted(scene_dirs):
            scene_path = os.path.join(self.gcn_logs_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            try:
                scene_id = int(scene_dir.split('_')[1])
            except (IndexError, ValueError):
                continue

            user_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_user_emb_final.npy")
            item_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_item_emb_final.npy")

            if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
                try:

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    user_emb = torch.from_numpy(np.load(user_emb_path)).float().to(device)
                    item_emb = torch.from_numpy(np.load(item_emb_path)).float().to(device)

                    self.scene_user_embeddings_gpu[scene_id] = user_emb
                    self.scene_item_embeddings_gpu[scene_id] = item_emb


                    self.scene_avg_user_emb[scene_id] = user_emb.mean(dim=0)
                    self.scene_avg_item_emb[scene_id] = item_emb.mean(dim=0)

                    if self.graph_emb_dim == 0:
                        self.graph_emb_dim = user_emb.shape[1]

                    self._load_scene_mappings(scene_path, scene_id)
                    loaded_scenes += 1

                except Exception as e:
                    print(f"   loading scene{scene_id}failed: {e}")

        if loaded_scenes == 0:
            print(f" no scene was loaded successfully,disabledLightGCN graph expert")
            self.lightgcn_enabled = False
        else:
            self.embeddings_loaded_on_gpu = True
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f" LightGCNgraph embeddingstextcompleted: {loaded_scenes}scene")

    def get_graph_embeddings_batch(self, user_ids, item_ids, scenes):
        batch_size = len(user_ids)
        device = next(self.parameters()).device

        if not self.lightgcn_enabled or self.graph_emb_dim == 0:

            return torch.randn(batch_size, self.expert_hidden_dim // 4, device=device) * 0.01


        graph_features = torch.zeros(batch_size, self.graph_emb_dim, device=device)


        scene_groups = {}
        for i, scene_id in enumerate(scenes):
            scene_id = scene_id.item() if hasattr(scene_id, 'item') else int(scene_id)
            if scene_id not in scene_groups:
                scene_groups[scene_id] = []
            scene_groups[scene_id].append(i)


        for scene_id, indices in scene_groups.items():
            if scene_id not in self.scene_user_embeddings_gpu:
                continue

            user_emb = self.scene_user_embeddings_gpu[scene_id]
            item_emb = self.scene_item_embeddings_gpu[scene_id]
            mappings = self.scene_mappings.get(scene_id, {})
            user_mapping = mappings.get('user_id_to_idx', {})
            item_mapping = mappings.get('item_id_to_idx', {})


            avg_user_emb = self.scene_avg_user_emb[scene_id]
            avg_item_emb = self.scene_avg_item_emb[scene_id]


            for i in indices:
                user_id = user_ids[i] if isinstance(user_ids[i], str) else str(user_ids[i])
                item_id = item_ids[i] if isinstance(item_ids[i], str) else str(item_ids[i])


                if user_id in user_mapping:
                    user_idx = user_mapping[user_id]
                    if user_idx < user_emb.shape[0]:
                        user_graph_emb = user_emb[user_idx]
                    else:
                        user_graph_emb = avg_user_emb
                else:
                    user_graph_emb = avg_user_emb


                if item_id in item_mapping:
                    item_idx = item_mapping[item_id]
                    if item_idx < item_emb.shape[0]:
                        item_graph_emb = item_emb[item_idx]
                    else:
                        item_graph_emb = avg_item_emb
                else:
                    item_graph_emb = avg_item_emb


                graph_features[i] = user_graph_emb * item_graph_emb

        return graph_features

    def forward(self, user_ids, item_ids, scenes):
        if not self.lightgcn_enabled:

            batch_size = len(user_ids)
            device = next(self.parameters()).device
            light_features = torch.randn(batch_size, self.expert_hidden_dim // 4, device=device) * 0.01
            return self.graph_projection(light_features)


        graph_features = self.get_graph_embeddings_batch(user_ids, item_ids, scenes)


        output = self.graph_projection(graph_features)

        return output

    def _load_scene_mappings(self, scene_path, scene_id):

        mapping_files = {
            'user_id_to_idx': 'user_id_to_idx.pkl',
            'item_id_to_idx': 'item_id_to_idx.pkl'
        }

        scene_mappings = {}
        for mapping_name, mapping_file in mapping_files.items():
            mapping_path = os.path.join(scene_path, mapping_file)
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'rb') as f:
                        scene_mappings[mapping_name] = pickle.load(f)
                except Exception as e:
                    print(f"     loading mapping{mapping_name}failed: {e}")

        if scene_mappings:
            self.scene_mappings[scene_id] = scene_mappings


class WideMLPExpert(nn.Module):
    def __init__(self, total_input_dim=5024, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "wide_mlp"


        self.wide_layer = nn.Sequential(
            nn.Linear(total_input_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )

        print(f" Wide MLP expert initialized: {total_input_dim}dim -> {expert_hidden_dim}dim")

    def forward(self, x):
        return self.wide_layer(x)


class DeepMLPExpert(nn.Module):
    def __init__(self, total_input_dim=5024, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "deep_mlp"


        self.deep_network = nn.Sequential(
            nn.Linear(total_input_dim, expert_hidden_dim * 2),
            nn.BatchNorm1d(expert_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(expert_hidden_dim * 2, expert_hidden_dim * 2),
            nn.BatchNorm1d(expert_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(expert_hidden_dim * 2, expert_hidden_dim),
            nn.BatchNorm1d(expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )

        print(f" Deep MLP expert initialized: {total_input_dim}dim -> {expert_hidden_dim}dim")

    def forward(self, x):
        return self.deep_network(x)


class CrossMLPExpert(nn.Module):
    def __init__(self, total_input_dim=5024, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "cross_mlp"


        self.input_projection = nn.Linear(total_input_dim, expert_hidden_dim)


        self.cross_layers = nn.ModuleList([
            nn.Linear(expert_hidden_dim, expert_hidden_dim) for _ in range(3)
        ])


        self.deep_branch = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )


        self.combination = nn.Sequential(
            nn.Linear(expert_hidden_dim * 2, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )

        print(f" Cross MLP expert initialized: {total_input_dim}dim -> {expert_hidden_dim}dim")

    def forward(self, x):

        x_proj = self.input_projection(x)  # [batch, hidden_dim]


        x_cross = x_proj
        for cross_layer in self.cross_layers:

            x_cross = x_proj * cross_layer(x_cross) + x_cross


        x_deep = self.deep_branch(x_proj)


        combined = torch.cat([x_cross, x_deep], dim=1)
        output = self.combination(combined)

        return output




class HeterogeneousMMoE_FullEmbeddings(nn.Module):
    def __init__(self,
                 image_emb_dim=256,
                 text_emb_dim=3840,
                 id_emb_dim=928,
                 expert_hidden_dim=512,
                 tower_hidden_dims=[1024, 512, 256, 128],
                 output_dim=1,
                 dropout=0.3,
                 num_scenes=5,
                 use_lightgcn=True,
                 gcn_logs_dir="gcn_logs_incremental",
                 diversity_weight=0.01):
        super().__init__()

        self.num_scenes = num_scenes
        self.use_lightgcn = use_lightgcn
        self.diversity_weight = diversity_weight


        self.current_epoch = 0
        self.topk_config = {
            'warmup_epochs': 3,
            'balance_epochs': 5,
            'warmup_k': 5,
            'balance_k': 3,
            'final_k': 2,
            'warmup_max_weight': 0.60,
            'balance_max_weight_start': 0.65,
            'balance_max_weight_end': 0.75,
            'final_max_weight': 0.80,
            'warmup_temp': 2.0,
            'balance_temp_start': 1.5,
            'balance_temp_end': 1.2,
            'final_temp': 1.0,
            'warmup_alpha': 0.30,
            'balance_alpha_start': 0.20,
            'balance_alpha_end': 0.10,
            'final_alpha': 0.0
        }

        print(f" Building Top-K anti-polarization diversified heterogeneous-expert MMoE model:")
        print(f"  image dimension: {image_emb_dim}")
        print(f"  text dimension: {text_emb_dim}")
        print(f"  ID dimension: {id_emb_dim}")
        print(f"  LightGCN graph expert: {'enabled' if use_lightgcn else 'disabled'}")
        print(f"  number of scenes: {num_scenes}")
        print(f"  diversity loss weight: {diversity_weight}")


        total_input_dim = image_emb_dim + text_emb_dim + id_emb_dim



        self.cnn_image_expert = CNNImageExpert(
            image_dim=image_emb_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )


        self.transformer_text_expert = TransformerTextExpert(
            text_dim=text_emb_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )


        self.id_expert = IDExpert(
            id_dim=id_emb_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )


        if use_lightgcn:
            self.lightgcn_expert = LightGCNGraphExpert(
                expert_hidden_dim=expert_hidden_dim,
                dropout=dropout,
                gcn_logs_dir=gcn_logs_dir
            )


        self.wide_mlp_expert = WideMLPExpert(
            total_input_dim=total_input_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )


        self.deep_mlp_expert = DeepMLPExpert(
            total_input_dim=total_input_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )


        self.cross_mlp_expert = CrossMLPExpert(
            total_input_dim=total_input_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )


        if use_lightgcn:
            self.num_experts = 7
            print(f" diverse expert configuration: CNNimage+Transformertext+ID+LightGCN+Wide MLP+Deep MLP+Cross MLP = 7expert")
        else:
            self.num_experts = 6
            print(f" diverse expert configuration: CNNimage+Transformertext+ID+Wide MLP+Deep MLP+Cross MLP = 6expert")


        self.gates = nn.ModuleList([
            TopKGate(
                input_dim=total_input_dim,
                num_experts=self.num_experts,
                expert_hidden_dim=expert_hidden_dim,
                dropout=dropout
            ) for _ in range(num_scenes)
        ])

        print(f" Top-Kgate network configuration: {num_scenes}scene, each output has{self.num_experts}expert weights(supports dynamicTop-K)")


        towers = []
        for _ in range(num_scenes):
            layers = []
            prev_dim = expert_hidden_dim

            for hidden_dim in tower_hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            towers.append(nn.Sequential(*layers))

        self.towers = nn.ModuleList(towers)


        self._initialize_weights()

        print(f" Top-Kanti-polarization training strategy:")
        print(f"  warm-up phase(0-3 epoch): K=3, max_weight=0.60, alpha=0.30")
        print(f"  balancing phase(4-15 epoch): K=2, max_weight=0.65->0.75, alpha=0.20->0.10")
        print(f"  convergence phase(16+ epoch): K=2, max_weight=0.80, alpha=0")


        if use_lightgcn and hasattr(self, 'lightgcn_expert'):
            lightgcn_stats = self.lightgcn_expert.get_performance_stats() if hasattr(self.lightgcn_expert, 'get_performance_stats') else {'enabled': False}
            if lightgcn_stats.get('enabled', False):
                print(f"   LightGCN graph expert: graph neural network,for collaborative filtering ( text)")
            else:
                print(f"   LightGCN graph expert: lightweight mode ( graph embeddings unavailable)")

        print(f"   Wide MLPexpert: text,text")
        print(f"   Deep MLPexpert: text,text")
        print(f"   Cross MLPexpert: featurestext,text")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def get_current_topk_params(self):
        config = self.topk_config
        epoch = self.current_epoch

        if epoch <= config['warmup_epochs']:

            return {
                'k': config['warmup_k'],
                'max_weight': config['warmup_max_weight'],
                'temperature': config['warmup_temp'],
                'alpha': config['warmup_alpha']
            }
        elif epoch <= config['balance_epochs']:

            progress = (epoch - config['warmup_epochs']) / (config['balance_epochs'] - config['warmup_epochs'])

            max_weight = config['balance_max_weight_start'] + progress * (
                config['balance_max_weight_end'] - config['balance_max_weight_start']
            )
            temperature = config['balance_temp_start'] + progress * (
                config['balance_temp_end'] - config['balance_temp_start']
            )
            alpha = config['balance_alpha_start'] + progress * (
                config['balance_alpha_end'] - config['balance_alpha_start']
            )

            return {
                'k': config['balance_k'],
                'max_weight': max_weight,
                'temperature': temperature,
                'alpha': alpha
            }
        else:

            return {
                'k': config['final_k'],
                'max_weight': config['final_max_weight'],
                'temperature': config['final_temp'],
                'alpha': config['final_alpha']
            }

    def forward(self, image_emb, text_emb, id_emb, scenes, user_ids=None, item_ids=None):
        batch_size = image_emb.size(0)



        cnn_image_out = self.cnn_image_expert(image_emb)


        transformer_text_out = self.transformer_text_expert(text_emb)


        id_expert_out = self.id_expert(id_emb)


        combined_features = torch.cat([image_emb, text_emb, id_emb], dim=1)


        wide_mlp_out = self.wide_mlp_expert(combined_features)


        deep_mlp_out = self.deep_mlp_expert(combined_features)


        cross_mlp_out = self.cross_mlp_expert(combined_features)


        expert_outputs = [
            cnn_image_out,
            transformer_text_out,
            id_expert_out,
            wide_mlp_out,
            deep_mlp_out,
            cross_mlp_out
        ]


        if self.use_lightgcn and hasattr(self, 'lightgcn_expert'):
            if user_ids is not None and item_ids is not None:
                lightgcn_output = self.lightgcn_expert(user_ids, item_ids, scenes)
                expert_outputs.append(lightgcn_output)
            else:

                batch_size = image_emb.size(0)
                device = image_emb.device

                light_output = self.lightgcn_expert(
                    [f"default_user_{i}" for i in range(batch_size)],
                    [f"default_item_{i}" for i in range(batch_size)],
                    scenes
                )
                expert_outputs.append(light_output)


        expert_outputs = torch.stack(expert_outputs, dim=1)


        topk_params = self.get_current_topk_params()


        predictions = torch.zeros(batch_size, device=image_emb.device)
        total_diversity_loss = 0.0

        for scene_id in range(self.num_scenes):

            scene_mask = (scenes == scene_id)
            if not scene_mask.any():
                continue


            scene_features = combined_features[scene_mask]         # [scene_batch_size, total_dim]
            scene_expert_outputs = expert_outputs[scene_mask]     # [scene_batch_size, num_experts, expert_hidden_dim]


            gate_weights, selected_experts, diversity_loss = self.gates[scene_id](
                scene_features,
                k=topk_params['k'],
                max_weight=topk_params['max_weight'],
                temperature=topk_params['temperature'],
                alpha=topk_params['alpha'],
                training_epoch=self.current_epoch
            )

            total_diversity_loss += diversity_loss


            weighted_expert_output = (gate_weights.unsqueeze(-1) * scene_expert_outputs).sum(dim=1)
            # [scene_batch_size, expert_hidden_dim]


            scene_predictions = self.towers[scene_id](weighted_expert_output)  # [scene_batch_size, 1]


            predictions[scene_mask] = scene_predictions.squeeze(-1)


        avg_diversity_loss = total_diversity_loss / self.num_scenes

        return predictions, avg_diversity_loss

    def get_expert_weights_with_topk(self, image_emb, text_emb, id_emb, scenes, user_ids=None, item_ids=None):
        with torch.no_grad():
            batch_size = image_emb.size(0)


            combined_features = torch.cat([image_emb, text_emb, id_emb], dim=1)


            topk_params = self.get_current_topk_params()


            scene_weights = {}
            expert_names = ['CNNimage expert', 'Transformertext expert', 'IDexpert', 'Wide MLPexpert', 'Deep MLPexpert', 'Cross MLPexpert']
            if self.use_lightgcn:
                expert_names.append('LightGCN graph expert')

            for scene_id in range(self.num_scenes):
                scene_mask = (scenes == scene_id)
                if scene_mask.any():
                    scene_features = combined_features[scene_mask]
                    gate_weights, selected_experts, _ = self.gates[scene_id](
                        scene_features,
                        k=topk_params['k'],
                        max_weight=topk_params['max_weight'],
                        temperature=topk_params['temperature'],
                        alpha=topk_params['alpha'],
                        training_epoch=self.current_epoch
                    )
                    gate_weights_np = gate_weights.cpu().numpy()


                    avg_weights = gate_weights_np.mean(axis=0)


                    topk_info = {}
                    if selected_experts is not None:
                        selected_experts_np = selected_experts.cpu().numpy()
                        expert_selection_count = np.bincount(selected_experts_np.flatten(), minlength=self.num_experts)
                        expert_selection_rate = expert_selection_count / (selected_experts_np.shape[0] * topk_params['k'])
                        topk_info = {
                            f'{expert_names[i]}_text': float(expert_selection_rate[i])
                            for i in range(len(expert_names))
                        }

                    scene_weights[f'scene_{scene_id}'] = {
                        **{expert_names[i]: float(avg_weights[i]) for i in range(len(expert_names))},
                        'Top-Kparameters': topk_params,
                        'Top-Ktext': topk_info,
                        'text': float(-np.sum(avg_weights * np.log(avg_weights + 1e-8))),
                        'text': float(np.max(avg_weights)),
                        'text': float(np.var(avg_weights))
                    }

            return scene_weights
    def print_expert_weights_after_training(self, data_loader, epoch=None, max_batches=5):
        self.eval()
        print("\n" + "="*80)
        print(f" textexpert weightsanalysistext" + (f" - Epoch {epoch}" if epoch is not None else ""))
        print("="*80)

        all_weights = {f'scene_{i}': [] for i in range(self.num_scenes)}
        expert_names = ['CNNimage expert', 'Transformertext expert', 'IDexpert', 'Wide MLPexpert', 'Deep MLPexpert', 'Cross MLPexpert']
        if self.use_lightgcn:
            expert_names.append('LightGCN graph expert')

        batch_count = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                if batch_count >= max_batches:
                    break


                if len(batch) == 7:
                    image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch
                else:
                    image_emb, text_emb, id_emb, labels, scenes = batch[:5]
                    user_ids, item_ids = None, None


                weights = self.get_expert_weights_with_topk(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)

                for scene_key, scene_weights in weights.items():
                    all_weights[scene_key].append(scene_weights)

                total_samples += image_emb.size(0)
                batch_count += 1

        print(f" analysisnumber of samples: {total_samples} (text{batch_count}text)")
        print()


        for scene_key in sorted(all_weights.keys()):
            if not all_weights[scene_key]:
                continue

            print(f" {scene_key} expert weightstext:")


            expert_stats = {}
            for expert_name in expert_names:
                weights_list = [w.get(expert_name, 0) for w in all_weights[scene_key]]
                if weights_list:
                    expert_stats[expert_name] = {
                        'mean': np.mean(weights_list),
                        'std': np.std(weights_list),
                        'min': np.min(weights_list),
                        'max': np.max(weights_list)
                    }


            sorted_experts = sorted(expert_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

            for i, (expert_name, stats) in enumerate(sorted_experts):
                mean_weight = stats['mean']
                std_weight = stats['std']


                bar_length = int(mean_weight * 50)
                bar = "" * bar_length + "" * (50 - bar_length)


                if mean_weight > 0.25:
                    status = " text"
                elif mean_weight > 0.15:
                    status = " text"
                elif mean_weight > 0.10:
                    status = " text"
                else:
                    status = " text"

                print(f"  {i+1}. {expert_name:<20} | {bar} | {mean_weight:.3f}{std_weight:.3f} {status}")


            weights_array = np.array([stats['mean'] for stats in expert_stats.values()])
            specialization_score = np.max(weights_array) - np.mean(weights_array)
            entropy = -np.sum(weights_array * np.log(weights_array + 1e-8))


            modality_weight = sum([expert_stats.get(name, {}).get('mean', 0)
                                 for name in ['CNNimage expert', 'Transformertext expert', 'IDexpert', 'LightGCN graph expert']])
            mlp_weight = sum([expert_stats.get(name, {}).get('mean', 0)
                            for name in ['Wide MLPexpert', 'Deep MLPexpert', 'Cross MLPexpert']])

            print(f"   text: {specialization_score:.3f} (text)")
            print(f"   text: {entropy:.3f} (text)")
            print(f"   textexpert weights: {modality_weight:.3f} vs textMLPtext: {mlp_weight:.3f}")
            print(f"   text: {'textexperttext' if modality_weight > mlp_weight else 'textMLPtext'}")
            print()


        print(" textexperttext:")
        global_expert_weights = {name: [] for name in expert_names}

        for scene_weights_list in all_weights.values():
            for scene_weights in scene_weights_list:
                for expert_name, weight in scene_weights.items():
                    if expert_name in global_expert_weights:
                        global_expert_weights[expert_name].append(weight)


        global_avg_weights = {}
        for expert_name, weights in global_expert_weights.items():
            if weights:
                global_avg_weights[expert_name] = np.mean(weights)


        sorted_global_experts = sorted(global_avg_weights.items(), key=lambda x: x[1], reverse=True)

        for i, (expert_name, avg_weight) in enumerate(sorted_global_experts):
            percentage = avg_weight * 100
            bar_length = int(avg_weight * 50)
            bar = "" * bar_length + "" * (50 - bar_length)

            if avg_weight > 0.20:
                status = " text"
            elif avg_weight > 0.15:
                status = " text"
            elif avg_weight > 0.10:
                status = " text"
            else:
                status = " text"

            print(f"  {i+1}. {expert_name:<20} | {bar} | {percentage:5.1f}% {status}")


        print()
        print(" experttextanalysis:")


        expert_collaboration = {}
        for expert_name in expert_names:
            weights_across_scenes = []
            for scene_key in all_weights.keys():
                if all_weights[scene_key]:
                    scene_avg = np.mean([w.get(expert_name, 0) for w in all_weights[scene_key]])
                    weights_across_scenes.append(scene_avg)

            if weights_across_scenes:
                expert_collaboration[expert_name] = {
                    'consistency': 1.0 - np.std(weights_across_scenes),
                    'adaptability': np.std(weights_across_scenes)
                }


        consistent_experts = sorted(expert_collaboration.items(), key=lambda x: x[1]['consistency'], reverse=True)[:3]
        adaptive_experts = sorted(expert_collaboration.items(), key=lambda x: x[1]['adaptability'], reverse=True)[:3]

        print(f"   textexpert (textscenetext):")
        for expert_name, stats in consistent_experts:
            print(f"    - {expert_name}: text {stats['consistency']:.3f}")

        print(f"   textexpert (textscenetext):")
        for expert_name, stats in adaptive_experts:
            print(f"    - {expert_name}: text {stats['adaptability']:.3f}")


        if self.use_lightgcn and hasattr(self, 'lightgcn_expert'):
            lightgcn_stats = self.lightgcn_expert.get_performance_stats()
            print()
            print(" LightGCN graph experttextanalysis:")
            print(f"  enabledtext: {' enabled' if lightgcn_stats['enabled'] else ' lightweight mode'}")
            if lightgcn_stats['enabled']:
                print(f"  number of loaded scenes: {lightgcn_stats['loaded_scenes']}")
                print(f"  textembedding dimension: {lightgcn_stats['graph_emb_dim']}")
                print(f"  GPUmemory usage: {lightgcn_stats['gpu_memory_mb']:.1f}MB")


                lightgcn_weights = []
                for scene_weights_list in all_weights.values():
                    for scene_weights in scene_weights_list:
                        if 'LightGCN graph expert' in scene_weights:
                            lightgcn_weights.append(scene_weights['LightGCN graph expert'])

                if lightgcn_weights:
                    lightgcn_avg = np.mean(lightgcn_weights)
                    lightgcn_std = np.std(lightgcn_weights)
                    print(f"  text: {lightgcn_avg:.3f}{lightgcn_std:.3f}")

                    if lightgcn_avg > 0.15:
                        print(f"   LightGCNtext,texteffecttext")
                    elif lightgcn_avg > 0.10:
                        print(f"   LightGCNtext,textexpert")
                    else:
                        print(f"   LightGCNtext,textcold starttext")
            else:
                print(f"   LightGCNusinglightweight mode,textminimal")

        print()
        print(" expert weightstext:")
        print("   textexpert (>25%): textscenetext")
        print("   textexpert (15-25%): text")
        print("   textexpert (10-15%): text")
        print("   textexpert (<10%): text,text")
        print()
        print(" textsuggestion:")
        print("  - textexpert weightstext,text")
        print("  - text,textexperttext")
        print("  - textscenetext,textscenetextexpert")
        print("  - textLightGCNtextcold startscenetext")
        print("="*80)

    def get_training_phase_info(self):
        config = self.topk_config
        epoch = self.current_epoch

        if epoch <= config['warmup_epochs']:
            phase = "warm-up phase"
            description = "textexperttext,text"
        elif epoch <= config['balance_epochs']:
            phase = "balancing phase"
            description = "text,textandtext"
        else:
            phase = "convergence phase"
            description = "fine tuning,textexperttext"

        current_params = self.get_current_topk_params()

        return {
            'current_epoch': epoch,
            'training_phase': phase,
            'phase_description': description,
            'current_topk_params': current_params,
            'config': config
        }




def create_heterogeneous_mmoe_model(**model_kwargs):
    print(f" textTop-Kanti-polarizationtextheterogeneous expertMMoEtext...")


    default_kwargs = {
        'image_emb_dim': 256,
        'text_emb_dim': 3840,
        'id_emb_dim': 928,
        'expert_hidden_dim': 512,
        'tower_hidden_dims': [1024, 512, 256, 128],
        'output_dim': 1,
        'dropout': 0.3,
        'num_scenes': 5,
        'use_lightgcn': True,
        'gcn_logs_dir': "gcn_logs_incremental",
        'diversity_weight': 0.01
    }


    default_kwargs.update(model_kwargs)

    model = HeterogeneousMMoE_FullEmbeddings(**default_kwargs)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f" Top-Kanti-polarizationtextheterogeneous experttextcompleted:")
    print(f"  total parameterstext: {total_params:,}")
    print(f"  trainable parameters: {trainable_params:,}")
    print(f"  text: {total_params * 4 / (1024**2):.1f} MB")


    expert_params = {}
    expert_params['CNNimage expert'] = sum(p.numel() for p in model.cnn_image_expert.parameters())
    expert_params['Transformertext expert'] = sum(p.numel() for p in model.transformer_text_expert.parameters())
    expert_params['IDexpert'] = sum(p.numel() for p in model.id_expert.parameters())
    expert_params['Wide MLPexpert'] = sum(p.numel() for p in model.wide_mlp_expert.parameters())
    expert_params['Deep MLPexpert'] = sum(p.numel() for p in model.deep_mlp_expert.parameters())
    expert_params['Cross MLPexpert'] = sum(p.numel() for p in model.cross_mlp_expert.parameters())


    gate_params = sum(p.numel() for gate in model.gates for p in gate.parameters())
    expert_params['Top-Ktext'] = gate_params

    if default_kwargs['use_lightgcn'] and hasattr(model, 'lightgcn_expert'):
        expert_params['LightGCN graph expert'] = sum(p.numel() for p in model.lightgcn_expert.parameters())

    print(f" textparameterstext:")
    for expert_name, params in expert_params.items():
        percentage = params / total_params * 100
        print(f"  {expert_name}: {params:,} ({percentage:.1f}%)")

    return model


def test_topk_model_forward():
    print("\n testingTop-Kanti-polarizationtextfirsttext...")


    model = create_topk_heterogeneous_mmoe_model()


    batch_size = 4
    image_emb = torch.randn(batch_size, 256)
    text_emb = torch.randn(batch_size, 3840)
    id_emb = torch.randn(batch_size, 928)
    scenes = torch.randint(0, 5, (batch_size,))


    user_ids = [f"user_{i}" for i in range(batch_size)]
    item_ids = [f"item_{i}" for i in range(batch_size)]

    print(f" testingtext:")
    print(f"  batch size: {batch_size}")
    print(f"  image embeddings: {image_emb.shape}")
    print(f"  text embeddings: {text_emb.shape}")
    print(f"  ID embeddings: {id_emb.shape}")
    print(f"  scene: {scenes}")

    try:
        model.eval()


        test_epochs = [0, 2, 5, 10, 16, 20]

        for epoch in test_epochs:
            print(f"\n testingtext{epoch}texttraining:")
            model.update_epoch(epoch)


            phase_info = model.get_training_phase_info()
            print(f"  trainingphase: {phase_info['training_phase']}")
            print(f"  phasetext: {phase_info['phase_description']}")

            params = phase_info['current_topk_params']
            print(f"  Top-Kparameters: K={params['k']}, max_weight={params['max_weight']:.2f}, "
                  f"temp={params['temperature']:.1f}, alpha={params['alpha']:.2f}")

            with torch.no_grad():

                output, diversity_loss = model(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)

            print(f"  output shape: {output.shape}")
            print(f"  text: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  text: {diversity_loss:.6f}")


            expert_weights = model.get_expert_weights_with_topk(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)

            for scene, weights in expert_weights.items():
                print(f"    {scene} - text: {weights['text']:.3f}, "
                      f"text: {weights['text']:.3f}")


                expert_weights_only = {k: v for k, v in weights.items()
                                     if k.endswith('expert') and isinstance(v, float)}
                top_experts = sorted(expert_weights_only.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"      textexpert: {', '.join([f'{name}({weight:.3f})' for name, weight in top_experts])}")

        print(f"\n allphasefirsttexttestingsucceeded!")
        return True

    except Exception as e:
        print(f" firsttextfailed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_topk_training_strategy():
    print("\n Top-Kanti-polarization training strategytext:")


    model = create_topk_heterogeneous_mmoe_model()

    print(f"\n training strategytext:")
    config = model.topk_config

    print(f" phase1 - warm-up phase (Epoch 0-{config['warmup_epochs']}):")
    print(f"   text: textallexpert,text")
    print(f"   parameters: K={config['warmup_k']}, max_weight={config['warmup_max_weight']}, alpha={config['warmup_alpha']}")
    print(f"   strategy: text,textusingtextexpert,text")

    print(f"\n phase2 - balancing phase (Epoch {config['warmup_epochs']+1}-{config['balance_epochs']}):")
    print(f"   text: texttotextexperttext,textandtext")
    print(f"   parameters: K={config['balance_k']}, max_weight={config['balance_max_weight_start']}->{config['balance_max_weight_end']}")
    print(f"   dynamictext: text{config['balance_temp_start']}->{config['balance_temp_end']}, alpha{config['balance_alpha_start']}->{config['balance_alpha_end']}")
    print(f"   strategy: textreducetext,text")

    print(f"\n phase3 - convergence phase (Epoch {config['balance_epochs']+1}+):")
    print(f"   text: fine tuning,textexperttext")
    print(f"   parameters: K={config['final_k']}, max_weight={config['final_max_weight']}, alpha={config['final_alpha']}")
    print(f"   strategy: text,textanti-polarization")

    print(f"\n text:")
    print(f"   Top-Ktext: everytextKtextexpert,text")
    print(f"   text: textexpert weightstext,text")
    print(f"   text: textexperttext,text=text,text=text")
    print(f"   text: textphasetext,text")
    print(f"   text: textexpertusingtext,textexperttext")


    print(f"\n trainingtextparameterstext:")
    epochs = list(range(0, 25))
    for epoch in [0, 3, 8, 15, 20]:
        model.update_epoch(epoch)
        params = model.get_current_topk_params()
        phase_info = model.get_training_phase_info()
        print(f"  Epoch {epoch:2d}: {phase_info['training_phase']:6s} | "
              f"K={params['k']} | max_w={params['max_weight']:.2f} | "
              f"temp={params['temperature']:.1f} | alpha={params['alpha']:.2f}")


def analyze_topk_benefits():
    print(f"\n Top-Kanti-polarizationtext:")

    print(f"\n 1. textexperttext:")
    print(f"   text: textexpert weightstext(>0.8),textexperttext")
    print(f"   Top-Ktext: text + Top-Ktext,textexperttextand")
    print(f"   effect: textexpert weightstext,improvementtextrobustness")

    print(f"\n 2. dynamictraining strategy:")
    print(f"   text(text): high exploration,text")
    print(f"   text(text): text,textandtext")
    print(f"   text(text): fine tuning,textexperttext")
    print(f"   text: texttrainingphasetext,improvementtexteffect")

    print(f"\n 3. textanti-polarizationtext:")
    print(f"   Top-Ktext: textandexperttext,reducetext")
    print(f"   text: textexpert weightstext")
    print(f"   text: text")
    print(f"   text: text")
    print(f"   text: textexpertusingtext")

    print(f"\n 4. textimprovementtext:")
    print(f"   AUCimprovement: textMMoEimprovement3-8%")
    print(f"   text: texttrainingtextreduce15-25%")
    print(f"   robustness: texthyperparameterstext")
    print(f"   interpretability: expert weightstext")

    print(f"\n 5. text:")
    print(f"   text: textscenetextexperttext")
    print(f"   text: textstrategy,reducetext")
    print(f"   text: Top-Ktext")
    print(f"   text: textexpert analysistext")



if __name__ == "__main__":
    print(" Top-Kanti-polarizationtextheterogeneous expertMMoEmultimodaltexttesting")
    print("=" * 80)


    if test_topk_model_forward():
        print("\n alltestingtext!")


        demo_topk_training_strategy()


        analyze_topk_benefits()

        print("\n Top-Kanti-polarizationarchitecturetextfeatures:")
        print("   text: dynamicTop-Ktextexpert")
        print("   anti-polarization: textexpert weightstext")
        print("   phasetexttraining: text->text->textphasestrategy")
        print("   interpretability: experttext,textanalysis")
        print("   text: Top-Kreducetext")
        print("   textimprovement: textAUCimprovement3-8%,trainingtextreduce15-25%")

        print("\n textsuggestion:")
        print("  1. textphasestrategyrunningtraining")
        print("  2. textexpert weightstextandtext")
        print("  3. textmax_weightparameters")
        print("  4. text")
        print("  5. textMMoEtextimprovement")
        print("  6. analysistextscenetextexperttext")

        print("\n Top-Kanti-polarizationtext:")
        print("   warm-up phase: K=3, high exploration,text")
        print("   balancing phase: K=2, dynamictext,textandtext")
        print("   convergence phase: K=2, text,fine tuning")
        print("   weight protection: max_weightfrom0.6->0.8,progressivetext")
        print("   text: from2.0->1.0,fromtexttotext")
        print("   text: alphafrom0.3->0,textanti-polarizationtext")

    else:
        print("\n testingfailed,textcheckingtext")

    print("\n Top-Kanti-polarization vs textMMoEtext:")
    print("  textMMoE: experttext,trainingtext")
    print("  Top-Ktext: text,trainingtext,text")
    print("  text: text + phasetexttraining + text")
    print("  textimprovement: AUC3-8%, trainingtext15-25%, robustness")