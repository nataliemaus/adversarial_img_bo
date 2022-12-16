import sys 
sys.path.append("../")
import argparse 
from utils.adversarial_objective import AdversarialsObjective  
import math 
import numpy as np 
import pandas as pd 

def save_data(keys, all_losses, save_path):
    keys_arr = np.array(keys)
    losses_arr = np.array(all_losses)
    df = pd.DataFrame() 
    df['token'] = keys_arr[0:len(losses_arr)] 
    df[f"cat_loss"] = losses_arr
    df.to_csv(save_path, index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--bsz', type=int, default=3)
    parser.add_argument('--save_every', type=int, default=3)
    parser.add_argument('--save_path', default="single_token_cat_losses.csv")
    args = parser.parse_args() 
    objective = AdversarialsObjective(
        n_tokens=1,
        minimize=True, 
        batch_size=args.bsz,
        use_fixed_latents=False,
        project_back=True,
        avg_over_N_latents=2,
        allow_cat_prompts=False,
    )
    vocab = objective.tokenizer.get_vocab() 
    keys = list(vocab.keys())
    n_batches = math.ceil(len(keys)/args.bsz)
    all_losses = [] 
    for i in range(n_batches):
        prompts = keys[i*args.bsz:(i+1)*args.bsz] 
        out_dict = objective.pipeline(
                input_type="prompt",
                input_value=prompts, 
                output_types=['loss'],
                fixed_latents=None
            )
        losses = out_dict['loss'].tolist() 
        all_losses = all_losses + losses
        if i % args.save_every == 0:
            save_data(keys, all_losses, args.save_path)
    save_data(keys, all_losses, args.save_path)
    import pdb 
    pdb.set_trace() 
    # pip install pandas 
    # python3 find_cat_tokens.py --bsz 20 --save_every 100 