# main.py
import numpy as np
import torch
import torch.nn.functional as F
import random
import os
from torch.utils.data import DataLoader





def contrastive_loss_sym(a, o, T=0.25):
    a = F.normalize(a, dim=-1)
    o = F.normalize(o, dim=-1)
    logits = a @ o.t() / T
    labels = torch.arange(a.size(0), device=a.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss1 + loss2)
seed = 2
def build_batch(batch, region_embeddings, od_embeddings, train_idx_map):
    origins = []
    origin_idx = []
    dst_embeds = []
    dst_distances = []
    true_flows = []
    ori_embeds = []
    ori_od_embeds = []
    dst_od_embeds = []

    for idx, sample in enumerate(batch):
        ori = sample['origin']
        dst = sample['destinations']
        prob = sample['prob']
        distances = sample['distances']

        # region embeddings
        ori_embed = region_embeddings[ori]  # [D]
        dst_embed = region_embeddings[dst]  # [len_dst, D]
        ori_od_embed = od_embeddings[train_idx_map[ori]]  # [od_D]

        # od embeddings (只在train_idx_map中找)
        dst_od_embed = []
        for d in dst:
            if d in train_idx_map:  # 只对训练区域有效
                dst_od_embed.append(od_embeddings[train_idx_map[d]])
            else:
                dst_od_embed.append(torch.zeros(od_embeddings.shape[1]).cuda())  # padding 0
        dst_od_embed = torch.stack(dst_od_embed, dim=0)  # [len_dst, od_dim]

        flow_gt = torch.FloatTensor(prob).cuda()
        dist_tensor = torch.FloatTensor(distances).unsqueeze(1).cuda()  # [N, 1]
        
        origins.extend([ori] * len(distances))
        origin_idx.extend([idx] * len(distances))
        dst_embeds.append(dst_embed.unsqueeze(0))
        dst_od_embeds.append(dst_od_embed.unsqueeze(0))
        dst_distances.append(dist_tensor.unsqueeze(0))
        true_flows.append(flow_gt)
        ori_embeds.append(ori_embed.repeat(len(distances), 1).unsqueeze(0))  # [N, D
        ori_od_embeds.append(ori_od_embed.repeat(len(distances), 1).unsqueeze(0))  # [N, D]

    origin_idx = torch.LongTensor(origin_idx).cuda()
    ori_embeds = torch.cat(ori_embeds, dim=0).cuda()          # [total_pairs, region_dim]
    ori_od_embeds = torch.cat(ori_od_embeds, dim=0).cuda()    # [total_pairs, od_dim]
    dst_embeds = torch.cat(dst_embeds, dim=0).cuda()          # [total_pairs, region_dim]
    dst_od_embeds = torch.cat(dst_od_embeds, dim=0).cuda()    # [total_pairs, od_dim]
    dst_distances = torch.cat(dst_distances, dim=0).cuda()    # [total_pairs, 1]
    true_flows = torch.cat(true_flows).cuda()                 # [total_pairs]

    return origin_idx, ori_embeds, dst_embeds, dst_od_embeds, true_flows, dst_distances,ori_od_embeds


def set_seed(args):
    if "SF" in args.city_path:
        args.seed = seed    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def cal_regression_metrics(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    cpc = (2 * np.sum(np.minimum(y_pred, y_true))) / (np.sum(y_pred) + np.sum(y_true) + 1e-8)

    eps = 1e-8
    den = np.sqrt(np.mean((y_true - np.mean(y_true))**2)) + eps   # std (population)
    nrmse_std = rmse / den

    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'NRMSE': nrmse_std, 'CPC': cpc}

def evaluate_valid(samples, region_embeddings , distributepredictor):
    loader = DataLoader(samples, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    preds, gts = [], []
    total_samples = 0
    for batch in loader:
        for sample in batch:
            ori = sample['origin']
            destinations = sample['destinations']
            probs = sample['prob']
            distances = sample['distances']

            ori_embed = region_embeddings[ori].unsqueeze(0)
            dst_embed = region_embeddings[destinations]
            dis_embed = torch.FloatTensor(distances).unsqueeze(1).cuda()
            ori_embed = ori_embed.repeat(dst_embed.shape[0], 1)

            pair_embed = torch.cat([ori_embed, dst_embed, dis_embed], dim=-1)
            flow1 = distributepredictor(pair_embed).view(-1, 1)

            flow_pred = flow1.squeeze()
            flow_gt = torch.FloatTensor(probs).cuda()

            preds.append(flow_pred.detach().cpu().numpy())
            gts.append(flow_gt.cpu().numpy())
            total_samples += len(flow_gt)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    print(f"Total samples evaluated: {total_samples}")
    return cal_regression_metrics(preds, gts)




def evaluate(samples, region_embeddings, distributepredictor ):
    loader = DataLoader(samples, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    preds, gts = [], []
    total_samples = 0
    for batch in loader:
        for sample in batch:
            ori = sample['origin']
            destinations = sample['destinations']
            probs = sample['prob']
            distances = sample['distances']

            ori_embed = region_embeddings[ori].unsqueeze(0)
            dst_embed = region_embeddings[destinations]
            dis_embed = torch.FloatTensor(distances).unsqueeze(1).cuda()
            ori_embed = ori_embed.repeat(dst_embed.shape[0], 1)

            pair_embed = torch.cat([ori_embed, dst_embed, dis_embed], dim=-1)
            flow1 = distributepredictor(pair_embed).view(-1, 1)

            flow_pred = flow1.squeeze()
            flow_gt = torch.FloatTensor(probs).cuda()

            preds.append(flow_pred.detach().cpu().numpy())
            gts.append(flow_gt.cpu().numpy())
            total_samples += len(flow_gt)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    print(f"Total samples evaluated: {total_samples}")
    return cal_regression_metrics(preds, gts)


