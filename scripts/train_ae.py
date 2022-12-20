import numpy as np
import sys 
sys.path.append("../")
import torch 
import argparse 
import wandb 
import os 
os.environ["WANDB_SILENT"] = "true" 
from torch.utils.data import TensorDataset, DataLoader
from utils.adversarial_objective import AdversarialsObjective
from utils.autoencoder import AE 

def start_wandb(args):
    args_dict = vars(args)
    tracker = wandb.init(
        entity=args_dict['wandb_entity'], 
        project=args_dict['wandb_project_name'],
        config=args_dict, 
    ) 
    print('running', wandb.run.name) 
    return tracker 

def train(args):
    ae = AE(
        input_shape=768,
        n_layers=args.n_layers,
    ) 
    if args.load_ckpt_wandb_name: 
        path_to_state_dict = f"../ae_models/{args.load_ckpt_wandb_name}.pkl" 
        state_dict = torch.load(path_to_state_dict) # load state dict 
        ae.load_state_dict(state_dict, strict=True) 
    ae = ae.cuda() 
    ae = ae.train() 
    objective = AdversarialsObjective(
        exclude_all_related_prompts=False,
        exclude_some_related_prompts=False,
    ) 
    all_embeddings = objective.all_token_embeddings.to(torch.float32).detach().cpu() # .numpy()
    tracker = start_wandb(args)
    optimizer = torch.optim.Adam([{'params': ae.parameters(), 'lr': args.lr} ], lr=args.lr)
    criterion = torch.nn.MSELoss()
    train_dataset = TensorDataset(all_embeddings.cuda(), all_embeddings.cuda() )
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True)
    model_save_path = f"../ae_models/{wandb.run.name}.pkl"
    lowest_loss = torch.inf 
    for e in range(args.n_epochs):
        losses_for_epoch = []
        for (inputs, scores) in train_loader:
            output = ae(inputs.cuda())
            loss = criterion(output, scores.cuda()) # .sum() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_for_epoch.append(loss.item() )
            tracker.log({"loss":loss.item()}) 
        avg_loss = np.array(losses_for_epoch).mean().item() 
        tracker.log({"avg_loss":avg_loss, "epoch":e})
        if avg_loss < lowest_loss:
            lowest_loss = avg_loss 
            tracker.log({"lowest_loss":lowest_loss })
            torch.save(ae.state_dict(), model_save_path) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--wandb_entity', default="nmaus" ) 
    parser.add_argument('--wandb_project_name', default="adversarial-bo-ae" )  
    parser.add_argument('--n_epochs', type=int, default=200_000_000_000_000_000_000) 
    parser.add_argument('--load_ckpt_wandb_name', default="" ) 
    # args 
    parser.add_argument('--lr', type=float, default=0.001 )  
    parser.add_argument('--bsz', type=int, default=128)  
    parser.add_argument('--n_layers', type=int, default=5)  
    args = parser.parse_args() 
    # torch.Size([49408, 768]) = Ntokens x 768 
    train(args)
    # CUDA_VISIBLE_DEVICES=0 python3 simple_ae.py --load_ckpt_wandb_name sage-firefly-13 --lr 0.0001 --n_layers 5
