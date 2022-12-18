
import torch
import gpytorch
import sys 
sys.path.append("../")
from gpytorch.mlls import PredictiveLogLikelihood 
from utils.bo_utils.trust_region import TrustRegionState, generate_batch, update_state
from utils.bo_utils.ppgpr import GPModelDKL 
from torch.utils.data import TensorDataset, DataLoader
from utils.adversarial_objective import AdversarialsObjective  
import argparse 
import wandb 
import math 
import os 
os.environ["WANDB_SILENT"] = "true" 
import random 

def initialize_global_surrogate_model(init_points, hidden_dims):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
    model = GPModelDKL(
        init_points.cuda(), 
        likelihood=likelihood,
        hidden_dims=hidden_dims,
    ).cuda()
    model = model.eval() 
    model = model.cuda()
    return model  

def start_wandb(args_dict):
    tracker = wandb.init(
        entity=args_dict['wandb_entity'], 
        project=args_dict['wandb_project_name'],
        config=args_dict, 
    ) 
    print('running', wandb.run.name) 
    return tracker 

def update_surr_model(
    model,
    learning_rte,
    train_z,
    train_y,
    n_epochs
):
    model = model.train() 
    mll = PredictiveLogLikelihood(model.likelihood, model, num_data=train_z.shape[0] )
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rte} ], lr=learning_rte)
    train_bsz = min(len(train_y),128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda()).sum() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()
    return model

def get_init_prompts(args):
    single_token_prompts = ["apple", "road", "ocean", "chair", "hat", "store", "knife", "moon", "red", "music"]
    single_token_prompts = single_token_prompts[0:args.bsz]
    N = args.n_tokens - 1
    if args.prepend_to_text:
        N = args.n_tokens 
    prompts = []
    for i in range(len(single_token_prompts)):
        prompt = ""
        for j in range(N):
            if j > 0: 
                prompt += " "
            prompt += random.choice(single_token_prompts)
        prompts.append(prompt)

    return prompts 

def get_init_data(args, prompts, objective):
    YS = [] 
    XS = [] 
    # if do batches of more than 10, get OOM 
    n_batches = math.ceil(args.n_init_pts / len(prompts)) 
    for i in range(n_batches): 
        X = objective.get_init_word_embeddings(prompts) 
        X = X.detach().cpu()  
        X = X.reshape(len(prompts), objective.dim ) 
        if i > 0: 
            X = X + torch.randn(X.shape)* 0.01 # multiply by 0.01 since  word_embed are typically small 
        XS.append(X)   
        print(X.shape) 
        YS.append(objective(X.to(torch.float16))) 
    Y = torch.cat(YS).detach().cpu() 
    X = torch.cat(XS).detach().cpu() 
    Y = Y.unsqueeze(-1)  
    return X, Y

def save_stuff(args, X, Y, objective, tracker):
    best_x = X[Y.argmax(), :].squeeze().to(torch.float16)
    pass_in_x = torch.cat([best_x.unsqueeze(0)]*args.bsz)
    imgs, xs, y = objective.query_oracle(pass_in_x, return_img=True)
    best_imgs = imgs[0] 
    if type(best_imgs) != list:
        best_imgs = [best_imgs]
    torch.save(best_x, f"../best_xs/{wandb.run.name}-best-x.pt") 
    for im_ix, img in enumerate(best_imgs):
        img.save(f"../best_xs/{wandb.run.name}_im{im_ix}.png")
    if objective.project_back: 
        best_prompt = xs[0] 
        tracker.log({"best_prompt":best_prompt}) 

def optimize(args):
    objective = AdversarialsObjective(
        n_tokens=args.n_tokens,
        minimize=args.minimize, 
        batch_size=args.bsz,
        use_fixed_latents=args.use_fixed_latents,
        project_back=args.project_back,
        avg_over_N_latents=args.avg_over_N_latents,
        allow_related_prompts=args.allow_related_prompts,
        seed=args.seed,
        prepend_to_text=args.prepend_to_text,
        optimal_class=args.optimal_class,
        visualize=False,
    )
    
    args_dict = vars(args)
    tracker = start_wandb(args_dict)
    tr = TrustRegionState(dim=objective.dim)
    assert objective.dim == args.n_tokens*768 

    # random sequence of n_tokens of these is each init prompt 
    prompts = get_init_prompts(args)
    X, Y = get_init_data(args, prompts, objective)
    model = initialize_global_surrogate_model(X, hidden_dims=args.hidden_dims) 
    model = update_surr_model(
        model=model,
        learning_rte=args.lr,
        train_z=X,
        train_y=Y,
        n_epochs=args.init_n_epochs
    )
    prev_best = args.threshold_save_best # before about this we don't care to log imgs, etc. s
    while objective.num_calls < args.max_n_calls:
        tracker.log({
            'num_calls':objective.num_calls,
            'best_y':Y.max(),
            'best_x':X[Y.argmax(), :].squeeze().tolist(), 
        } ) 
        if Y.max().item() > prev_best or args.debug: 
            prev_best = Y.max().item() 
            save_stuff(args, X, Y, objective, tracker)
        elif (prev_best == args.threshold_save_best) and (objective.num_calls > 3_000):
            # if we still don't exceed -1.6 after 10k calls, start recording any progress at all 
            prev_best = -torch.inf 
        x_next = generate_batch( 
            state=tr,
            model=model,
            X=X,
            Y=Y,
            batch_size=args.bsz, 
            acqf=args.acq_func,
            absolute_bounds=(objective.lb, objective.ub)
        ) 
        y_next = objective(x_next.to(torch.float16) ).unsqueeze(-1)
        update_state(tr, y_next)
        Y = torch.cat((Y, y_next.detach().cpu()), dim=-2) 
        X = torch.cat((X, x_next.detach().cpu()), dim=-2) 
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
    parser.add_argument('--max_n_calls', type=int, default=200_000_000_000_000_000_000) 
    parser.add_argument('--lr', type=float, default=0.01 ) 
    parser.add_argument('--n_epochs', type=int, default=2)  
    parser.add_argument('--version', type=int, default=4)  
    parser.add_argument('--init_n_epochs', type=int, default=80) 
    parser.add_argument('--acq_func', type=str, default='ts' ) 
    parser.add_argument('--debug', type=bool, default=False)  
    parser.add_argument('--minimize', type=bool, default=True)  
    parser.add_argument('--use_fixed_latents', type=bool, default=False)  
    parser.add_argument('--project_back', type=bool, default=True)  
    parser.add_argument('--allow_related_prompts', type=bool, default=False)  
    parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)") 
    parser.add_argument('--avg_over_N_latents', type=int, default=5)
    parser.add_argument('--threshold_save_best', type=int, default=-4)
    ## modify... 
    parser.add_argument('--seed', type=int, default=1 ) 
    parser.add_argument('--bsz', type=int, default=10)  
    parser.add_argument('--prepend_task', type=bool, default=False)
    parser.add_argument('--prepend_to_text', default="a picture of a dog") 
    parser.add_argument('--n_tokens', type=int, default=3 )  
    parser.add_argument('--optimal_class', default="cat" )  
    args = parser.parse_args() 
    if not args.prepend_task: # if default task, prepend_to_text = ""
        args.prepend_to_text = ""
    assert args.minimize 
    assert args.version == 4
    if args.debug:
        args.n_init_pts = 10
        args.init_n_epochs = 2 
        args.bsz = 5
        args.max_n_calls = 100
        args.avg_over_N_latents = 3 
    assert args.n_init_pts % args.bsz == 0
    optimize(args)

    # python3 --prepend_task True --n_tokens 3 

    # pip install diffusers
    # pip install accelerate 
    #  conda activate lolbo_mols
    # tmux attach -t adv 

    # conda create --name adv_env --file adv_env.txt
    # conda activate adv_env
    # gauss node 2! 
    # tmux attach -t adv1 , adv2, adv3, adv4
    # CUDA_VISIBLE_DEVICES=1 python3 optimize.py --n_tokens 12 --prepend_task True
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --n_tokens 6 --prepend_task True
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --n_tokens 14 --prepend_task True
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --n_tokens 7 --prepend_task True
    # tmux attach -t adv10 (node1)
    # CUDA_VISIBLE_DEVICES=0 python3 optimize.py --n_tokens 4 --prepend_task True

    # Allegro 
    # tmux attach -t adv adv2, adv3, adv4, adv5, adv6, adv7 
    # CUDA_VISIBLE_DEVICES=0 python3 optimize.py --n_tokens 20 --prepend_task True --bsz 20
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --n_tokens 16 --prepend_task True --bsz 20
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --n_tokens 10 --prepend_task True --bsz 20
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --n_tokens 6 --prepend_task True --bsz 20 --seed 3
    # CUDA_VISIBLE_DEVICES=5 python3 optimize.py --n_tokens 8 --prepend_task True --bsz 20 --seed 3
    # CUDA_VISIBLE_DEVICES=6 python3 optimize.py --n_tokens 50 --prepend_task True --bsz 20
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --n_tokens 12 --prepend_task True --bsz 20

    # moving xs from desktop to jkgardner: 
    # rsync -a --ignore-existing best_xs jkgardner.com:/home/nmaus/adversarial_img_bo/

    # Up Next::: ,    conda activate adv_env
    # gauss node 1, tmux attach -t adv11, adv12, adv13
    # CUDA_VISIBLE_DEVICES=6 python3 optimize.py --n_tokens 4 --bsz 10 --seed 1 --optimal_class car 
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --n_tokens 6 --bsz 10 --seed 1 --optimal_class car 
    # CUDA_VISIBLE_DEVICES=8 python3 optimize.py --n_tokens 8 --bsz 10 --seed 1 --optimal_class car
    # gauss node 2, tmux attach -t adv5 
    # CUDA_VISIBLE_DEVICES=9 python3 optimize.py --n_tokens 6 --bsz 5 --seed 1 --optimal_class violin
