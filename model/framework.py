import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfAttentionAggregator(nn.Module):
    def __init__(self, embed_dim, k):
        super(SelfAttentionAggregator, self).__init__()
        self.k = k
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, target_vecs, neighbor_vecs):
        """
        target_vecs: [num_nodes, embed_dim]
        neighbor_vecs: [num_nodes, k, embed_dim]
        mask: [num_nodes, k] (0/1), 仅对 mask=1 的邻居做注意力聚合
        return: [num_nodes, embed_dim]
        """
        Q = self.q_proj(target_vecs).unsqueeze(1)          # [num_nodes, 1, embed_dim]
        K = self.k_proj(neighbor_vecs)                     # [num_nodes, k, embed_dim]
        V = self.v_proj(neighbor_vecs)                     # [num_nodes, k, embed_dim]

        # 计算注意力分数
        attn_scores = (Q * K).sum(dim=-1) / (K.size(-1) ** 0.5)  # [num_nodes, k]

        # 使用 mask 将无效位置的分数设为非常小
        # attn_scores = attn_scores.masked_fill(mask == 0, -1e9)   # [num_nodes, k]

        # softmax 得到注意力权重（屏蔽位置几乎为0）
        attn_weights = self.softmax(attn_scores)                 # [num_nodes, k]

        # 注意力加权求和
        weighted_sum = (attn_weights.unsqueeze(-1) * V).sum(dim=1)  # [num_nodes, embed_dim]
        return weighted_sum

    
class SpatialAttentionModel(nn.Module):
    def __init__(self, num_nodes, feat_dim, embed_dim=60, k=30):
        super(SpatialAttentionModel, self).__init__()
        self.k = k
        self.embed_dim = embed_dim

        # 区域原始特征编码（可选）
        self.feat_encoder = nn.Linear(embed_dim, embed_dim)

        # 每个节点初始化流入和流出向量（需要训练）
        self.in_embed = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.out_embed = nn.Parameter(torch.randn(num_nodes, embed_dim))

        # 自注意力聚合器
        self.in_agg = SelfAttentionAggregator(embed_dim, k)
        self.out_agg = SelfAttentionAggregator(embed_dim, k)

    def forward(self, node_feat, neighbors_index):
        """
        node_feat: [num_nodes, feat_dim]
        neighbors_index: [num_nodes, k] 每个区域的k个最近邻索引
        """
        node_feat = node_feat  # [num_nodes, embed_dim]

        in_vecs = self.in_embed
        out_vecs = self.out_embed
        # print(f"in_vecs: {in_vecs.shape}, out_vecs: {out_vecs.shape}, node_feat: {node_feat.shape}, neighbors_index: {neighbors_index.shape}")

        # 获取邻居向量
        neighbor_in_vecs = in_vecs[neighbors_index]     # [num_nodes, k, embed_dim]
        neighbor_out_vecs = out_vecs[neighbors_index]   # [num_nodes, k, embed_dim]

        # 聚合邻居
        agg_in = self.in_agg(in_vecs, neighbor_in_vecs)     # [num_nodes, embed_dim]
        agg_out = self.out_agg(out_vecs, neighbor_out_vecs) # [num_nodes, embed_dim]

        # 可以将输出拼接或加权求和再进入下一层
        final_rep = torch.cat([agg_in,agg_out, node_feat], dim=1)  # [num_nodes, embed_dim * 3]
        
        return final_rep
    
    def get_role_embeddings(self, node_feat, neighbors_index):
        """
        返回每个区域的 emissive 和 attractive 嵌入:
          e_E: [N, D]  emissive (outgoing / source-like) embedding
          e_A: [N, D]  attractive (incoming / sink-like) embedding
        """
        # 和 forward 一样的邻居聚合过程，只是不再拼接 node_feat
        in_vecs = self.in_embed
        out_vecs = self.out_embed

        neighbor_in_vecs = in_vecs[neighbors_index]      # [N, k, D]
        neighbor_out_vecs = out_vecs[neighbors_index]    # [N, k, D]

        agg_in = self.in_agg(in_vecs,  neighbor_in_vecs)     # [N, D] attractive
        agg_out = self.out_agg(out_vecs, neighbor_out_vecs)  # [N, D] emissive

        e_E = agg_out  # emissive embedding  (E for Emissive)
        e_A = agg_in   # attractive embedding (A for Attractive)

class SpatialAttentionModel(nn.Module):
    def __init__(self, num_nodes, feat_dim, embed_dim=60, k=30):
        super(SpatialAttentionModel, self).__init__()
        self.k = k
        self.embed_dim = embed_dim

        # 区域原始特征编码（可选）
        self.feat_encoder = nn.Linear(embed_dim, embed_dim)

        # 每个节点初始化流入和流出向量（需要训练）
        self.in_embed = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.out_embed = nn.Parameter(torch.randn(num_nodes, embed_dim))

        # 自注意力聚合器
        self.in_agg = SelfAttentionAggregator(embed_dim, k)
        self.out_agg = SelfAttentionAggregator(embed_dim, k)

    def forward(self, node_feat, neighbors_index):
        """
        node_feat: [num_nodes, feat_dim]
        neighbors_index: [num_nodes, k] 每个区域的k个最近邻索引
        """
        node_feat = node_feat  # [num_nodes, embed_dim]

        in_vecs = self.in_embed
        out_vecs = self.out_embed
        # print(f"in_vecs: {in_vecs.shape}, out_vecs: {out_vecs.shape}, node_feat: {node_feat.shape}, neighbors_index: {neighbors_index.shape}")

        # 获取邻居向量
        neighbor_in_vecs = in_vecs[neighbors_index]     # [num_nodes, k, embed_dim]
        neighbor_out_vecs = out_vecs[neighbors_index]   # [num_nodes, k, embed_dim]

        # 聚合邻居
        agg_in = self.in_agg(in_vecs, neighbor_in_vecs)     # [num_nodes, embed_dim]
        agg_out = self.out_agg(out_vecs, neighbor_out_vecs) # [num_nodes, embed_dim]

        # 可以将输出拼接或加权求和再进入下一层
        final_rep = torch.cat([agg_in,agg_out, node_feat], dim=1)  # [num_nodes, embed_dim * 3]
        
        return final_rep
    
    def get_role_embeddings(self, node_feat, neighbors_index):
        """
        返回每个区域的 emissive 和 attractive 嵌入:
          e_E: [N, D]  emissive (outgoing / source-like) embedding
          e_A: [N, D]  attractive (incoming / sink-like) embedding
        """
        # 和 forward 一样的邻居聚合过程，只是不再拼接 node_feat
        in_vecs = self.in_embed
        out_vecs = self.out_embed

        neighbor_in_vecs = in_vecs[neighbors_index]      # [N, k, D]
        neighbor_out_vecs = out_vecs[neighbors_index]    # [N, k, D]

        agg_in = self.in_agg(in_vecs,  neighbor_in_vecs)     # [N, D] attractive
        agg_out = self.out_agg(out_vecs, neighbor_out_vecs)  # [N, D] emissive

        e_E = agg_out  # emissive embedding  (E for Emissive)
        e_A = agg_in   # attractive embedding (A for Attractive)
        e = torch.cat([e_A, node_feat,e_E], dim=1)         # [N, 2D + feat_dim(或D)]
        return e_E, e_A
    
# class SpatialAttentionModel(nn.Module):
#     def __init__(self, num_nodes, feat_dim, embed_dim=60, k=30):
#         super(SpatialAttentionModel, self).__init__()
#         self.k = k
#         self.embed_dim = embed_dim

#         # 区域原始特征编码
#         self.feat_encoder = nn.Linear(feat_dim, embed_dim)

#         # 用于生成 in/out embedding 的线性层（替代随机初始化）
#         self.in_encoder = nn.Linear(feat_dim, embed_dim)
#         self.out_encoder = nn.Linear(feat_dim, embed_dim)

#         # 自注意力聚合器
#         self.in_agg = SelfAttentionAggregator(embed_dim, k)
#         self.out_agg = SelfAttentionAggregator(embed_dim, k)

#     def forward(self, node_feat, neighbors_index):
#         """
#         node_feat: [num_nodes, feat_dim]
#         neighbors_index: [num_nodes, k] 每个区域的k个最近邻索引
#         """
#         # 对节点特征进行编码
#         node_feat_enc = node_feat    # [num_nodes, embed_dim]

#         # 使用 node_feat 生成 in/out embedding
#         in_vecs = self.in_encoder(node_feat)             # [num_nodes, embed_dim]
#         out_vecs = self.out_encoder(node_feat)           # [num_nodes, embed_dim]

#         # 获取邻居向量
#         neighbor_in_vecs = in_vecs[neighbors_index]      # [num_nodes, k, embed_dim]
#         neighbor_out_vecs = out_vecs[neighbors_index]    # [num_nodes, k, embed_dim]

#         # 聚合邻居
#         agg_in = self.in_agg(in_vecs, neighbor_in_vecs)      # [num_nodes, embed_dim]
#         agg_out = self.out_agg(out_vecs, neighbor_out_vecs)  # [num_nodes, embed_dim]

#         # 最终拼接输出
#         final_rep = torch.cat([agg_in, agg_out, node_feat_enc], dim=1)  # [num_nodes, embed_dim * 3]
        
#         return final_rep
    
class DeepGravity(nn.Module):
    def __init__(self):
        super(DeepGravity, self).__init__()

        hiddim = 256 # 256
        layers = 3 # 15

        self.linear_in = nn.Linear(311, hiddim)
        self.linears = nn.ModuleList(
            [nn.Linear(hiddim, hiddim) for i in range(layers)]
        )
        self.linear_out = nn.Linear(hiddim, 1)

    def forward(self, input):
        input = self.linear_in(input)
        x = input
        for layer in self.linears:
            x = torch.relu(layer(x)) + x
        x = torch.tanh(self.linear_out(x))
        return x
    


class FlowPredictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(155, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward(self, x):  # x: [B, D]
        return self.mlp(x).squeeze(-1)  # [B]
    
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
            # nn.Dropout(0.1),
            nn.Linear(256,1)
        )

    def forward(self, x):  # x: [B, D]
        return self.relation_mlp(x).squeeze(-1)  # [B]
    
class OD_normer():
    def __init__(self, min_, max_):
        self.min_ = min_
        self.max_ = max_

    def normalize(self, x):
        """Scale a value or array of values to the range [-1, 1]."""
        return 2 * ((x - self.min_) / (self.max_ - self.min_)) - 1

    def renormalize(self, x):
        return ((x + 1) / 2) * (self.max_ - self.min_) + self.min_
    


class PairTransformer(nn.Module):
    def __init__(self, embed_dim, nhead=4, num_layers=2):
        super(PairTransformer, self).__init__()
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, pair_embed):
        """
        pair_embed: [B, 2D]
        return: flow_pred [B]
        """
        x = self.input_proj(pair_embed)  # [B, D]
        x = x.unsqueeze(0)  # [1, B, D] —— batch_first=True -> 视为 batch_size=1 的序列
        x = self.transformer(x)  # [1, B, D]
        x = x.squeeze(0)  # [B, D]
        flow_pred = x  # [B]
        return flow_pred
    


class DistancePositionalEncoding(nn.Module):
    def __init__(self, pe_dim):
        super().__init__()
        self.pe_dim = pe_dim
        self.mlp = nn.Sequential(
            nn.Linear(2, pe_dim),  # 输入2维：距离和距离排名
            nn.ReLU(),
            nn.Linear(pe_dim, pe_dim)
        )

    def forward(self, distances, ranks):
        """
        distances: Tensor [B, K] → origin 到每个 destination 的距离
        ranks: Tensor [B, K] → 每个 destination 在候选列表中的排名（从0到K-1）
        return: PE: [B, K, pe_dim]
        """
        pe_input = torch.stack([distances, ranks], dim=-1)  # [B, K, 2]
        pe = self.mlp(pe_input)  # [B, K, pe_dim]
        return pe
    

class SegmentTransformerGate(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),  # concat(origin, dest, distance)
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, pair_embed):
        """
        ori_embed: [B, D]
        dst_embed: [B, K, D]
        distances: [B, K]
        return: fused: [B, K, D]
        """
        # B, K, D = dst_embed.size()

        # # expand ori embedding
        # ori_exp = ori_embed.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
        # dist_input = distances.unsqueeze(-1)                # [B, K, 1]

        # # compute gate
        # gate_input = torch.cat([ori_exp, dst_embed, dist_input], dim=-1)  # [B, K, 2D+1]
        # gate = self.gate_mlp(gate_input)                                  # [B, K, D]

        # fused = gate * dst_embed + (1 - gate) * ori_exp                   # [B, K, D]

        # Transformer encoding
        out = self.attn(pair_embed)  # [B, K, D]

        return out
    
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
