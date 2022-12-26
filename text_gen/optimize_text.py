
import torch
import numpy as np 
import argparse 
import wandb 
import math 
import os 
import pandas as pd 
import sys 
sys.path.append("../")
from text_gen.text_gen_objective import AdversarialsTextGenObjective
os.environ["WANDB_SILENT"] = "true" 
from scripts.optimize import (
    RunTurbo,
    tuple_type,
)

class OptimizeText(RunTurbo):
    def __init__(self, args):
        self.args = args 

    def get_init_data(self,):
        YS = [] 
        XS = [] 
        PS = []
        GS = []
        # if do batches of more than 10, get OOM 
        n_batches = math.ceil(self.args.n_init_pts / self.args.bsz) 
        for _ in range(n_batches): 
            X = torch.randn(self.args.bsz, self.args.objective.dim )*0.01
            if self.args.single_number_per_token:
                X = torch.rand(self.args.bsz, self.args.objective.dim )
            XS.append(X)   
            prompts, ys, gen_text = self.args.objective(X.to(torch.float16))
            YS.append(ys) 
            PS = PS + prompts
            GS = GS + gen_text 
        Y = torch.cat(YS).detach().cpu() 
        self.args.X = torch.cat(XS).float().detach().cpu() 
        self.args.Y = Y.unsqueeze(-1)  
        self.args.P = PS
        self.args.G = GS 


    def save_stuff(self, tracker):
        X = self.args.X
        Y = self.args.Y
        P = self.args.P
        G = self.args.G 
        best_x = X[Y.argmax(), :].squeeze().to(torch.float16)
        torch.save(best_x, f"../best_xs/{wandb.run.name}-best-x.pt") 
        best_prompt = P[Y.argmax()]
        tracker.log({"best_prompt":best_prompt}) 
        best_gen_text = G[Y.argmax()]
        tracker.log({"best_gen_text":best_gen_text}) 
        save_path = f"../best_xs/{wandb.run.name}-all-data.csv"
        prompts_arr = np.array(P)
        loss_arr = Y.squeeze().detach().cpu().numpy() 
        gen_text_arr = np.array(G)  # (10, 5)  = N, n_gen_text 
        df = pd.DataFrame() 
        df['prompt'] = prompts_arr
        df["loss"] = loss_arr 
        for i in range(gen_text_arr.shape[-1]): 
            df[f"gen_text{i+1}"] = gen_text_arr[:,i] 
        df.to_csv(save_path, index=None)


    def call_oracle_and_update_next(self, x_next):
        p_next, y_next, g_next = self.args.objective(x_next.to(torch.float16))
        self.args.P = self.args.P + p_next # prompts 
        self.args.G = self.args.G + g_next # generated text 
        return y_next

    def init_objective(self,):
        self.args.objective = AdversarialsTextGenObjective(
            num_gen_seq=self.args.num_gen_seq,
            max_gen_length=self.args.max_gen_length,
            dist_metric=self.args.dist_metric, # "sq_euclidean",
            n_tokens=self.args.n_tokens,
            minimize=self.args.minimize, 
            batch_size=self.args.bsz,
            prepend_to_text=self.args.prepend_to_text,
            visualize=False,
            compress_search_space=self.args.compress_search_space,
            single_number_per_token=self.args.single_number_per_token,
            lb = self.args.lb,
            ub = self.args.ub,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--wandb_entity', default="nmaus" ) 
    parser.add_argument('--wandb_project_name', default="adversarial-bo-txt" )  
    parser.add_argument('--n_init_per_prompt', type=int, default=None ) 
    parser.add_argument('--n_init_pts', type=int, default=None) 
    parser.add_argument('--lr', type=float, default=0.01 ) 
    parser.add_argument('--n_epochs', type=int, default=2)  
    parser.add_argument('--init_n_epochs', type=int, default=80) 
    parser.add_argument('--acq_func', type=str, default='ts' ) 
    parser.add_argument('--debug', type=bool, default=False) 
    parser.add_argument('--minimize', type=bool, default=True)  
    parser.add_argument('--task', default="textgen") 
    parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)") 
    parser.add_argument('--more_hdims', type=bool, default=True) # for >8 tokens only 
    parser.add_argument('--seed', type=int, default=1 ) 
    parser.add_argument('--prepend_to_text', default="I am happy") 
    parser.add_argument('--success_value', type=int, default=8)  
    parser.add_argument('--break_after_success', type=bool, default=True)
    ## meh  
    parser.add_argument('--max_n_calls', type=int, default=40_000) 
    parser.add_argument('--n_addtional_evals', type=int, default=1_000) # evals afteer success
    parser.add_argument('--num_gen_seq', type=int, default=5 ) 
    parser.add_argument('--max_gen_length', type=int, default=10 ) 
    parser.add_argument('--dist_metric', default="sq_euclidean" )  
    parser.add_argument('--compress_search_space', type=bool, default=False )
     ## modify...  
    parser.add_argument('--bsz', type=int, default=10)  
    parser.add_argument('--n_tokens', type=int, default=6 )  
    parser.add_argument('--prepend_task', type=bool, default=False)
    parser.add_argument('--failure_tolerance', type=int, default=10 )  
    parser.add_argument('--success_tolerance', type=int, default=10 )  
    parser.add_argument('--single_number_per_token', type=bool, default=False )
    parser.add_argument('--additive_gp', type=bool, default=False)  

    args = parser.parse_args() 

    runner = OptimizeText(args)
    runner.optimize() 

    # pip install diffusers
    # pip install accelerate 
    #  conda activate lolbo_mols
    # tmux attach -t adv 
    # moving xs from desktop to jkgardner: 
    # rsync -a --ignore-existing best_xs jkgardner.com:/home/nmaus/adversarial_img_bo/
    # conda create --name adv_env --file adv_env.txt
    # conda activate adv_env
    # pip install nltk 
    # RUNNING:::::::  
