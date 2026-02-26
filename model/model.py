import torch
import torch.nn as nn


class SelfAttentionAggregator(nn.Module):
    def __init__(self, embed_dim, k):
        super().__init__()
        self.k = k
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, target_vecs, neighbor_vecs):
        Q = self.q_proj(target_vecs).unsqueeze(1)          # [N, 1, D]
        K = self.k_proj(neighbor_vecs)                     # [N, k, D]
        V = self.v_proj(neighbor_vecs)                     # [N, k, D]

        attn_scores = (Q * K).sum(dim=-1) / (K.size(-1) ** 0.5)  # [N, k]
        attn_weights = self.softmax(attn_scores)                 # [N, k]
        weighted_sum = (attn_weights.unsqueeze(-1) * V).sum(dim=1)  # [N, D]
        return weighted_sum


class SpatialAttentionModel(nn.Module):
    def __init__(self, num_nodes, feat_dim, embed_dim=60, k=30):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim

        self.feat_encoder = nn.Linear(embed_dim, embed_dim)

        self.in_embed = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.out_embed = nn.Parameter(torch.randn(num_nodes, embed_dim))

        self.in_agg = SelfAttentionAggregator(embed_dim, k)
        self.out_agg = SelfAttentionAggregator(embed_dim, k)

    def forward(self, node_feat, neighbors_index):
        in_vecs = self.in_embed
        out_vecs = self.out_embed

        neighbor_in_vecs = in_vecs[neighbors_index]     # [N, k, D]
        neighbor_out_vecs = out_vecs[neighbors_index]   # [N, k, D]

        agg_in = self.in_agg(in_vecs, neighbor_in_vecs)      # [N, D]
        agg_out = self.out_agg(out_vecs, neighbor_out_vecs)  # [N, D]

        final_rep = torch.cat([agg_in, agg_out, node_feat], dim=1)  # [N, 3D]
        return final_rep

    def get_role_embeddings(self, node_feat, neighbors_index):
        in_vecs = self.in_embed
        out_vecs = self.out_embed

        neighbor_in_vecs = in_vecs[neighbors_index]     # [N, k, D]
        neighbor_out_vecs = out_vecs[neighbors_index]   # [N, k, D]

        agg_in = self.in_agg(in_vecs, neighbor_in_vecs)      # [N, D]
        agg_out = self.out_agg(out_vecs, neighbor_out_vecs)  # [N, D]

        e_E = agg_out  # emissive
        e_A = agg_in   # attractive
        return e_E, e_A


class distributePredictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.relation_mlp = nn.Sequential(
            nn.Linear(337, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.relation_mlp(x).squeeze(-1)
    


class ODFeatureMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class FeatureMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)
    
class ODJointMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.mlp(x).squeeze(-1)  # [B]