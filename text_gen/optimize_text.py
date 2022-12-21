
import torch
import numpy as np 
import argparse 
import wandb 
import math 
import os 
import pandas as pd 
import sys 
sys.path.append("../")
from utils.bo_utils.trust_region import (
    TrustRegionState,
     generate_batch, 
     update_state
)
from text_gen.text_gen_objective import AdversarialsTextGenObjective
os.environ["WANDB_SILENT"] = "true" 
from scripts.optimize import (
    initialize_global_surrogate_model, 
    start_wandb,
    update_surr_model,
)

def get_init_data(args, objective):
    YS = [] 
    XS = [] 
    PS = []
    GS = []
    # if do batches of more than 10, get OOM 
    n_batches = math.ceil(args.n_init_pts / args.bsz) 
    for _ in range(n_batches): 
        X = torch.randn(args.bsz, objective.dim )*0.01
        XS.append(X)   
        prompts, ys, gen_text = objective(X.to(torch.float16))
        YS.append(ys) 
        PS = PS + prompts
        GS = GS + gen_text 
    Y = torch.cat(YS).detach().cpu() 
    X = torch.cat(XS).float().detach().cpu() 
    Y = Y.unsqueeze(-1)  
    return X, Y, PS, GS 

def save_stuff(X, Y, P, G, tracker):
    best_x = X[Y.argmax(), :].squeeze().to(torch.float16)
    torch.save(best_x, f"../best_xs/{wandb.run.name}-best-x.pt") 
    best_prompt = P[Y.argmax()]
    tracker.log({"best_prompt":best_prompt}) 
    best_gen_text = G[Y.argmax()]
    tracker.log({"best_gen_text":best_gen_text}) 
    save_path = f"../best_xs/{wandb.run.name}-all-data.csv"
    prompts_arr = np.array(P)
    loss_arr = Y.squeeze().detach().cpu().numpy() 
    gen_text_arr = np.array(G)
    df = pd.DataFrame() 
    df['prompt'] = prompts_arr
    df["loss"] = loss_arr 
    df["gen_text"] = gen_text_arr
    df.to_csv(save_path, index=None)

def optimize(args):
    if args.debug:
        args.n_init_pts = 10
        args.init_n_epochs = 2 
        args.bsz = 5
        args.max_n_calls = 60
    assert args.n_init_pts % args.bsz == 0
    if not args.prepend_task: 
        args.prepend_to_text = "" 
    if (args.n_tokens > 8) and args.more_hdims: # best cats and cars so far have n_tokens = 4, 6, and 8
        args.hidden_dims = tuple_type("(1024,256,128,64)") 
    assert args.minimize 
    objective = AdversarialsTextGenObjective(
        num_gen_seq=args.num_gen_seq,
        max_gen_length=args.max_gen_length,
        dist_metric=args.dist_metric, # "sq_euclidean",
        n_tokens=args.n_tokens,
        minimize=args.minimize, 
        batch_size=args.bsz,
        prepend_to_text=args.prepend_to_text,
        visualize=False,
        compress_search_space=args.compress_search_space,
    )
    tr = TrustRegionState(dim=objective.dim)
    # random sequence of n_tokens of these is each init prompt 
    X, Y, P, G = get_init_data(args, objective)
    model = initialize_global_surrogate_model(X, hidden_dims=args.hidden_dims) 
    model = update_surr_model(
        model=model,
        learning_rte=args.lr,
        train_z=X,
        train_y=Y,
        n_epochs=args.init_n_epochs
    )
    tracker = start_wandb(args)
    prev_best = -torch.inf 
    while objective.num_calls < args.max_n_calls:
        tracker.log({
            'num_calls':objective.num_calls,
            'best_y':Y.max(),
            'best_x':X[Y.argmax(), :].squeeze().tolist(), 
        } ) 
        if Y.max().item() > prev_best: 
            prev_best = Y.max().item() 
            save_stuff(X, Y, P, G, tracker)
        if args.break_after_success and (prev_best > args.success_value):
            # only give n_addtional_evals more calls 
            args.max_n_calls = objective.num_calls + args.n_addtional_evals 
            args.break_after_success = False 
        x_next = generate_batch( 
            state=tr,
            model=model,
            X=X,
            Y=Y,
            batch_size=args.bsz, 
            acqf=args.acq_func,
            absolute_bounds=(objective.lb, objective.ub)
        ) 
        p_next, y_next, gen_next = objective(x_next.to(torch.float16))
        y_next = y_next.unsqueeze(-1)
        update_state(tr, y_next)
        Y = torch.cat((Y, y_next.detach().cpu()), dim=-2) 
        X = torch.cat((X, x_next.detach().cpu()), dim=-2) 
        P = P + p_next
        G = G + gen_next 
        model = update_surr_model(
            model=model,
            learning_rte=args.lr,
            train_z=x_next,
            train_y=y_next, 
            n_epochs=args.n_epochs
        )
    tracker.finish() 

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--wandb_entity', default="nmaus" ) 
    parser.add_argument('--wandb_project_name', default="adversarial-bo" )  
    parser.add_argument('--n_init_pts', type=int, default=1100) 
    parser.add_argument('--lr', type=float, default=0.01 ) 
    parser.add_argument('--n_epochs', type=int, default=2)  
    parser.add_argument('--init_n_epochs', type=int, default=80) 
    parser.add_argument('--acq_func', type=str, default='ts' ) 
    parser.add_argument('--debug', type=bool, default=False) 
    parser.add_argument('--minimize', type=bool, default=True)  
    parser.add_argument('--task', default="textgen") 
    parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)") 
    parser.add_argument('--more_hdims', type=bool, default=True) # for >8 tokens only 
    ## modify...  
    parser.add_argument('--seed', type=int, default=1 ) 
    parser.add_argument('--bsz', type=int, default=10)  
    parser.add_argument('--prepend_to_text', default="I am happy") 
    parser.add_argument('--break_after_success', type=bool, default=False)
    parser.add_argument('--max_n_calls', type=int, default=100_000) 
    parser.add_argument('--success_value', type=int, default=-1)  
    parser.add_argument('--n_addtional_evals', type=int, default=3_000) 
    # fr
    parser.add_argument('--n_tokens', type=int, default=6 )  
    parser.add_argument('--prepend_task', type=bool, default=False)
    parser.add_argument('--compress_search_space', type=bool, default=False )
    parser.add_argument('--num_gen_seq', type=int, default=5 ) 
    parser.add_argument('--max_gen_length', type=int, default=20 ) 
    parser.add_argument('--dist_metric', default="sq_euclidean" ) 
    args = parser.parse_args() 

    if args.compress_search_space:
        args.hidden_dims = tuple_type("(64,64,32)") 

    optimize(args)
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

    # CUDA_VISIBLE_DEVICES=1 python3 optimize_text.py --n_tokens 4

