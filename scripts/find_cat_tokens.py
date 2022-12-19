import sys 
sys.path.append("../")
import argparse 
from utils.adversarial_objective import AdversarialsObjective  
import math 
import numpy as np 
import pandas as pd 
import glob 

def save_data(keys, all_losses, save_path):
    keys_arr = np.array(keys)
    losses_arr = np.array(all_losses)
    df = pd.DataFrame() 
    df['token'] = keys_arr[0:len(losses_arr)] 
    df["loss"] = losses_arr
    df.to_csv(save_path, index=None)


def load_data():
    df = pd.read_csv("../data/all_single_token_cat_losses.csv")
    tokens = df['token'].values
    losses = df['cat_loss'].values 
    return tokens, losses 

def compilte_data(): 
    # put together csvs into single sorted csv 
    optimal_class = "violin"
    data_files = glob.glob(f"single_token_{optimal_class}_losses_*.csv")
    # 49408 total tokens! 
    all_tokens = []
    all_losses = []
    dfs = [] 
    for file in data_files: dfs.append(pd.read_csv(file))
    for df in dfs: all_tokens = all_tokens + df['token'].values.tolist() 
    for df in dfs:  all_losses = all_losses + df['loss'].values.tolist()
    tokens_arr = np.array(all_tokens)
    losses_arr = np.array(all_losses)
    ind = np.argsort(losses_arr) # , -4)[-4:]
    sorted_tokens = tokens_arr[ind].tolist()
    sorted_losses = losses_arr[ind].tolist() 
    save_data(sorted_tokens, sorted_losses, f"all_single_token_{optimal_class}_losses.csv")
    # test_df = pd.read_csv(f"all_single_token_{optimal_class}_losses.csv")
    # tokens = test_df['token'].values
    # losses = test_df['loss'].values


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--bsz', type=int, default=3)
    parser.add_argument('--optimal_class', default="cat")
    parser.add_argument('--save_every', type=int, default=3)
    parser.add_argument('--start_ix', type=int, default=0)
    parser.add_argument('--stop_ix', type=int, default=None) 
    args = parser.parse_args() 
    objective = AdversarialsObjective(
        n_tokens=1,
        minimize=True, 
        batch_size=args.bsz,
        use_fixed_latents=True,
        project_back=False,
        avg_over_N_latents=2,
        allow_related_prompts=True,
        visualize=False,
        prepend_to_text="",
        optimal_class=args.optimal_class,
    )
    vocab = objective.tokenizer.get_vocab() 
    keys = list(vocab.keys()) 
    keys.sort() # make sure order is always the same!  
    if args.stop_ix is None:
        args.stop_ix = len(keys)
    keys = keys[args.start_ix:args.stop_ix] 
    save_path = f"single_token_{args.optimal_class}_losses_{args.start_ix}_to_{args.stop_ix}.csv" 
    n_batches = math.ceil(len(keys)/args.bsz) 
    all_losses = [] 
    for i in range(n_batches):
        prompts = keys[i*args.bsz:(i+1)*args.bsz] 
        out_dict = objective.pipeline(
            input_type="prompt",
            input_value=prompts, 
            output_types=['loss'],
            fixed_latents=objective.fixed_latents,
        )
        losses = out_dict['loss'].tolist() 
        all_losses = all_losses + losses
        if i % args.save_every == 0:
            save_data(keys, all_losses, save_path)
    save_data(keys, all_losses, save_path)
    import pdb 
    pdb.set_trace() 

    # tmux new -s adv14 
    # conda activate adv_env
    # conda activate adv_env
    # gauss node 1
    # CUDA_VISIBLE_DEVICES=3 python3 find_cat_tokens.py --bsz 10 --save_every 20 --start_ix 0 --optimal_class violin

