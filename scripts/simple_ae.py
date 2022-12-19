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

class AE(torch.nn.Module):
    def __init__(
        self,
        input_shape=768,
        n_layers=5,
    ):
        super().__init__()
        enc_layers = []
        dec_layers = []
        for i in range(n_layers):
            if i > 0:
                enc_layers.append(torch.nn.ReLU())
                dec_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Linear(input_shape//(2**i), input_shape//(2**(i+1))))
            dec_layers.append(torch.nn.Linear(input_shape//(2**(n_layers-i)), input_shape//(2**(n_layers-i-1))))
        self.encoder = torch.nn.Sequential(*enc_layers)
        self.decoder = torch.nn.Sequential(*dec_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    ).cuda() 
    ae = ae.train() 
    objective = AdversarialsObjective(
        allow_related_prompts=True
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
    # args 
    parser.add_argument('--lr', type=float, default=0.001 )  
    parser.add_argument('--bsz', type=int, default=128)  
    parser.add_argument('--n_layers', type=int, default=5)  
    args = parser.parse_args() 
    # torch.Size([49408, 768]) = Ntokens x 768 
    train(args)
