import time
import argparse
import numpy as np
import torch

from data_load import load_data
from model import (SpatialAttentionModel,distributePredictor,ODFeatureMLP,FeatureMLP,ODJointMLP)
from pprint import pprint
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tools import contrastive_loss, build_batch, evaluate, evaluate_valid,contrastive_loss_sym,set_seed
torch.cuda.set_device(3)  
import numpy as np
import copy




def main(args):
    print("\n  **Loading data...")
    train_samples, valid_samples, test_samples, neighbors_index, attr, train_od_features,train_attr,train_idx,valid_idx, test_idx = load_data(
        city_path=args.city_path,
        neighbors_k=args.neighbor_k
    )
    print(train_od_features.shape)
    num_nodes, feat_dim = attr.shape
    k = neighbors_index.shape[1]

    region_encoder = SpatialAttentionModel(num_nodes, feat_dim, embed_dim=args.embed_dim, k=k).cuda()
    # ✅ 保存 epoch0（训练开始前）
    state_epoch0 = copy.deepcopy(region_encoder.state_dict())
  
    distributepredictor = distributePredictor(args.embed_dim).cuda()

    joint_input_dim = args.embed_dim *4 +1
    od_joint_mlp = ODJointMLP(input_dim=joint_input_dim, hidden_dim=args.embed_dim, output_dim=1).cuda()
    
    od_mlp = ODFeatureMLP(input_dim=train_od_features.shape[1], hidden_dim=args.embed_dim, output_dim=args.embed_dim).cuda()
    
    optimizer = torch.optim.Adam(
        list(region_encoder.parameters()) +
        list(distributepredictor.parameters()) +
        list(od_mlp.parameters())+
        list(od_joint_mlp.parameters()),  # 新增
        lr=args.lr
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)

    attr_tensor = torch.FloatTensor(attr).cuda()
    train_attr_tensor=torch.FloatTensor(train_attr).cuda()
    f = torch.FloatTensor(train_attr).cuda()
    neighbor_tensor = torch.LongTensor(neighbors_index).cuda()
    train_od_tensor = torch.FloatTensor(train_od_features).cuda()

    best_valid_loss = np.inf
    patience_counter = args.patience

    print("\n  **Start training...")
    for epoch in range(args.max_epoch):
        epoch_start = time.time()   # ✅ start timing
        print(f"Epoch {epoch + 1}:", end=" | ")
        region_encoder.train()

        distributepredictor.train()
   
        od_mlp.train()

        train_loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
        epoch_losses = []
        train_idx_map = {area_id: i for i, area_id in enumerate(train_idx)}


        # ----------------- 训练循环 -----------------
        for batch in train_loader:
            region_embeddings = region_encoder(attr_tensor, neighbor_tensor)  # [N, D]
   
            od_embeddings = od_mlp(train_od_tensor)
   


            cl_loss_sym = contrastive_loss_sym(train_attr_tensor, od_embeddings)
  

            # Flow预测 loss
            _, ori_embed, dst_embed,dst_od_embed, true_flow, dis_embed,ori_od_embed = build_batch(batch, region_embeddings,od_embeddings, train_idx_map)
            pair_embed = torch.cat([ori_embed, dst_embed, dis_embed], dim=-1)
            od_pair_embed = torch.cat([ori_embed, dst_od_embed,dis_embed], dim=-1)
            flow1 = distributepredictor(pair_embed).view(-1, 1)
            weights = 1 
            flow_loss = torch.mean(weights * (flow1.squeeze() - true_flow) ** 2)


            flow2 = od_joint_mlp(od_pair_embed).view(-1, 1)
 
            flow_loss2 = torch.mean(weights * (flow2.squeeze() - true_flow) ** 2)


            loss = flow_loss  + 5 * cl_loss_sym + flow_loss2 # 可调权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start  # ✅ end timing
        steps = len(epoch_losses)
        print(f"train loss={train_loss:.6f}", end=" | ")
        print(f"time/epoch={epoch_time:.2f}s, time/step={epoch_time/max(steps,1):.3f}s", end=" | ")

        # ----------------- 验证 -----------------
        region_encoder.eval()
        distributepredictor.eval()
        od_mlp.eval()
        with torch.no_grad():
            region_embeddings_val = region_encoder(attr_tensor, neighbor_tensor)
            metrics = evaluate_valid(valid_samples, region_embeddings_val, distributepredictor)
            valid_loss = metrics['MSE']
            scheduler.step(valid_loss)
            print(f"valid MSE={metrics['MSE']:.6f}")

            if metrics['MSE'] < best_valid_loss:
                best_valid_loss = metrics['MSE']
                patience_counter = args.patience
                best_model_state = {
                    'region_encoder': region_encoder.state_dict(),
                    'od_mlp': od_mlp.state_dict()
                }
            else:
                patience_counter -= 1
                if patience_counter == 0:
                    print("Early stopping!")
                    break


    # ----------------- 测试 -----------------
    print("\n  **Evaluating on test set...")
    region_encoder.load_state_dict(best_model_state['region_encoder'])

    od_mlp.load_state_dict(best_model_state['od_mlp'])

    region_encoder.eval()
    distributepredictor.eval()

    od_mlp.eval()
    with torch.no_grad():
        region_embeddings_test = region_encoder(attr_tensor, neighbor_tensor)
        metrics = evaluate(test_samples, region_embeddings_test,  distributepredictor)
        pprint(metrics)

        with open("results.txt", "a") as f:
            f.write("Joint Training with Flow Prediction + OD Contrastive\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--city_path", type=str, default="./data/NYC")
    parser.add_argument("--neighbor_k", type=int, default=30)
    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    set_seed(args)
    main(args)
