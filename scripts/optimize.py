
import torch
import gpytorch
import numpy as np 
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
import pandas as pd 
from utils.imagenet_classes import load_imagenet


class RunTurbo():
    def __init__(self, args):
        self.args = args 

    def initialize_global_surrogate_model(self, init_points, hidden_dims):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        model = GPModelDKL(
            init_points.cuda(), 
            likelihood=likelihood,
            hidden_dims=hidden_dims,
        ).cuda()
        model = model.eval() 
        model = model.cuda()
        return model  

    def start_wandb(self):
        args_dict = vars(self.args) 
        tracker = wandb.init(
            entity=args_dict['wandb_entity'], 
            project=args_dict['wandb_project_name'],
            config=args_dict, 
        ) 
        print('running', wandb.run.name) 
        return tracker 

    def update_surr_model(
        self,
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


    def get_init_prompts(self, objective ):
        related_vocab = objective.related_vocab
        all_vocab = list(objective.vocab.keys()) 
        random.shuffle(all_vocab)
        starter_vocab = all_vocab[0:100]
        tmp = [] 
        for vocab_word in starter_vocab:
            if not vocab_word in related_vocab:
                tmp.append(vocab_word)
        starter_vocab = tmp 
        N = self.args.n_tokens - 1
        if self.args.prepend_to_text:
            N = self.args.n_tokens 
        prompts = [] 
        iters = math.ceil(self.args.n_init_pts/self.args.n_init_per_prompt) 
        for i in range(iters):
            prompt = ""
            for j in range(N):
                if j > 0: 
                    prompt += " "
                prompt += random.choice(starter_vocab)
            prompts.append(prompt)

        return prompts 

    def get_init_data(self ):
        prompts = self.get_init_prompts(self.args.objective )
        YS = [] 
        XS = [] 
        PS = []
        # if do batches of more than 10, get OOM 
        n_batches = math.ceil(self.args.n_init_pts / (self.args.bsz*self.args.n_init_per_prompt)) 
        for i in range(n_batches): 
            prompt_batch = prompts[i*self.args.bsz:(i+1)*self.args.bsz] 
            X = self.args.objective.get_init_word_embeddings(prompt_batch) 
            X = X.detach().cpu() 
            X = X.reshape(self.args.bsz, self.args.objective.dim ) 
            for j in range(self.args.n_init_per_prompt): # 10 randoms near each prompt ! 
                if j > 0:
                    X = X + torch.randn(self.args.bsz, self.args.objective.dim)*0.01
                XS.append(X)   
                xs, ys = self.args.objective(X.to(torch.float16))
                YS.append(ys) 
                PS = PS + xs 
        Y = torch.cat(YS).detach().cpu() 
        self.args.X = torch.cat(XS).float().detach().cpu() 
        self.args.Y = Y.unsqueeze(-1)  
        self.args.P = PS 

    def save_stuff(self, tracker):
        X = self.args.X
        Y = self.args.Y
        P = self.args.P 
        best_x = X[Y.argmax(), :].squeeze().to(torch.float16)
        torch.save(best_x, f"../best_xs/{wandb.run.name}-best-x.pt") 
        if self.args.objective.project_back: 
            best_prompt = P[Y.argmax()]
            tracker.log({"best_prompt":best_prompt}) 
        save_path = f"../best_xs/{wandb.run.name}-all-data.csv"
        prompts_arr = np.array(P)
        loss_arr = Y.squeeze().detach().cpu().numpy() 
        df = pd.DataFrame() 
        df['prompt'] = prompts_arr
        df["loss"] = loss_arr 
        df.to_csv(save_path, index=None)

        if False:
            pass_in_x = torch.cat([best_x.unsqueeze(0)]*args.bsz)
            imgs, xs, y = objective.query_oracle(pass_in_x, return_img=True)
            best_imgs = imgs[0] 
            if type(best_imgs) != list:
                best_imgs = [best_imgs]
            for im_ix, img in enumerate(best_imgs):
                img.save(f"../best_xs/{wandb.run.name}_im{im_ix}.png")
            if objective.project_back: 
                best_prompt = xs[0] 
                tracker.log({"best_prompt":best_prompt}) 


    def init_args(self):
        if self.args.debug:
            self.args.n_init_pts = 8
            self.args.init_n_epochs = 2 
            self.args.bsz = 2
            self.args.max_n_calls = 200
            self.args.avg_over_N_latents = 2
            self.args.n_init_per_prompt = 2
        assert self.args.n_init_pts % self.args.bsz == 0
        if not self.args.prepend_task: 
            self.args.prepend_to_text = ""
        if (self.args.n_tokens > 8) and self.args.more_hdims: # best cats and cars so far have n_tokens = 4, 6, and 8
            self.args.hidden_dims = tuple_type("(1024,256,128,64)") 

    def call_oracle_and_update_next(self, x_next):
        prompts_next, y_next = self.args.objective(x_next.to(torch.float16))
        self.args.P = self.args.P + prompts_next
        return y_next

    def init_objective(self):
        self.args.objective = AdversarialsObjective(
            n_tokens=self.args.n_tokens,
            minimize=self.args.minimize, 
            batch_size=self.args.bsz,
            use_fixed_latents=self.args.use_fixed_latents,
            project_back=self.args.project_back,
            avg_over_N_latents=self.args.avg_over_N_latents,
            exclude_all_related_prompts=self.args.exclude_all_related_prompts,
            exclude_some_related_prompts=self.args.exclude_some_related_prompts,
            seed=self.args.seed,
            prepend_to_text=self.args.prepend_to_text,
            optimal_class=self.args.optimal_class,
            visualize=False,
            compress_search_space=self.args.compress_search_space,
            remove_synonyms=self.args.remove_synonyms,
        )

    def optimize(self):
        self.init_args()  
        tracker = self.start_wandb()
        self.init_objective() 
        self.get_init_data() 
        model = self.initialize_global_surrogate_model(
            self.args.X, 
            hidden_dims=self.args.hidden_dims
        ) 
        model = self.update_surr_model(
            model=model,
            learning_rte=self.args.lr,
            train_z=self.args.X,
            train_y=self.args.Y,
            n_epochs=self.args.init_n_epochs
        )
        prev_best = -torch.inf 
        num_tr_restarts = 0 
        tr = TrustRegionState(dim=self.args.objective.dim)
        while self.args.objective.num_calls < self.args.max_n_calls:
            tracker.log({
                'num_calls':self.args.objective.num_calls,
                'best_y':self.args.Y.max(),
                'best_x':self.args.X[self.args.Y.argmax(), :].squeeze().tolist(), 
                'tr_length':tr.length,
                'num_tr_restarts':num_tr_restarts,
            } ) 
            if self.args.Y.max().item() > prev_best: 
                prev_best = self.args.Y.max().item() 
                # save_stuff(args, X, Y, P, args.objective, tracker)
                self.save_stuff(tracker)
            if self.args.break_after_success and (prev_best > self.args.success_value):
                # only give n_addtional_evals more calls 
                self.args.max_n_calls = self.args.objective.num_calls + self.args.n_addtional_evals 
                self.args.break_after_success = False 
            x_next = generate_batch( 
                state=tr,
                model=model,
                X=self.args.X,
                Y=self.args.Y,
                batch_size=self.args.bsz, 
                acqf=self.args.acq_func,
                absolute_bounds=(self.args.objective.lb, self.args.objective.ub)
            ) 
            self.args.X = torch.cat((self.args.X, x_next.detach().cpu()), dim=-2) 
            y_next = self.call_oracle_and_update_next(x_next)
            y_next = y_next.unsqueeze(-1)
            self.args.Y = torch.cat((self.args.Y, y_next.detach().cpu()), dim=-2) 
            update_state(tr, y_next) 
            if tr.restart_triggered:
                num_tr_restarts += 1
                tr = TrustRegionState(dim=self.args.objective.dim)
                model = self.initialize_global_surrogate_model(self.args.X, hidden_dims=self.args.hidden_dims) 
                model = self.update_surr_model(
                    model=model,
                    learning_rte=self.args.lr,
                    train_z=self.args.X,
                    train_y=self.args.Y,
                    n_epochs=self.args.init_n_epochs
                )
            else:
                model = self.update_surr_model(
                    model=model,
                    learning_rte=self.args.lr,
                    train_z=x_next,
                    train_y=y_next, 
                    n_epochs=self.args.n_epochs
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
    parser.add_argument('--lr', type=float, default=0.01 ) 
    parser.add_argument('--n_epochs', type=int, default=2)  
    parser.add_argument('--init_n_epochs', type=int, default=80) 
    parser.add_argument('--acq_func', type=str, default='ts' ) 
    parser.add_argument('--debug', type=bool, default=False) 
    parser.add_argument('--minimize', type=bool, default=True)  
    parser.add_argument('--use_fixed_latents', type=bool, default=False)  
    parser.add_argument('--project_back', type=bool, default=True)  
    parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)") 
    parser.add_argument('--avg_over_N_latents', type=int, default=5) 
    parser.add_argument('--more_hdims', type=bool, default=True) # for >8 tokens only 
    parser.add_argument('--seed', type=int, default=1 ) 
    #  meh
    parser.add_argument('--prepend_to_text', default="a picture of a dog") 
    parser.add_argument('--break_after_success', type=bool, default=True )
    parser.add_argument('--success_value', type=int, default=-1)  
    # maybe later 
    parser.add_argument('--max_n_calls', type=int, default=20_000) 
    parser.add_argument('--n_addtional_evals', type=int, default=3_000) 
    parser.add_argument('--compression_version', type=int, default=2) # 2 == "laced-snow-14" 
    ## bsz ...  
    parser.add_argument('--n_init_pts', type=int, default=200) 
    parser.add_argument('--n_init_per_prompt', type=int, default=10 ) 
    parser.add_argument('--bsz', type=int, default=10)  
    # i.e. --bsz 28 --n_init_pts 280 
    # exclude realted? 
    parser.add_argument('--exclude_some_related_prompts', type=bool, default=False) 
    parser.add_argument('--exclude_all_related_prompts', type=bool, default=False)  
    parser.add_argument('--remove_synonyms', type=bool, default=False) # removingg some --> remove synonyms? 
    # main 
    parser.add_argument('--n_tokens', type=int, default=4 )  
    parser.add_argument('--optimal_class', default="all" )  
    parser.add_argument('--prepend_task', type=bool, default=False)
    parser.add_argument('--start_ix', type=int, default=0 ) # start and stop imnet 
    parser.add_argument('--stop_ix', type=int, default=100 ) # start and stop imnet 
    parser.add_argument('--compress_search_space', type=bool, default=False )
    args = parser.parse_args() 

    if args.compress_search_space:
        args.hidden_dims = tuple_type("(32,32,16)") 

    if args.optimal_class == "all":
        imagenet_dict = load_imagenet()
        classes = list(imagenet_dict.keys())  # 583 
        for clas in classes[args.start_ix:args.stop_ix]:
            args.optimal_class = clas 
            runner = RunTurbo(args) 
            runner.optimize() 
    else:
        runner = RunTurbo(args) 
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

    # ERIC MACHINE: ... ???   (ssh nmaus@deep-a6000x8-1.seas.upenn.edu )
    #   tmux attach -t adv0, adv1, adv2, adv3, ..., adv7
    #   dockerd-rootless-setuptool.sh install
    #   systemctl --user start docker
    #   docker run -v /home1/n/nmaus/adversarial_img_bo/:/workspace/ --gpus all -it nmaus/advenv
    # CUDA_VISIBLE_DEVICES=0 python3 optimize.py --n_tokens 2 --compress_search_space True --optimal_class goldfish --max_n_calls 40000 --n_init_pts 280 --bsz 28
    # CUDA_VISIBLE_DEVICES=1 python3 optimize.py --n_tokens 2 --optimal_class goldfish --max_n_calls 40000 --n_init_pts 280 --bsz 28
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --n_tokens 3 --compress_search_space True --optimal_class goldfish --max_n_calls 40000 --n_init_pts 280 --bsz 28
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --n_tokens 3 --optimal_class goldfish --max_n_calls 40000 --n_init_pts 280 --bsz 28
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --n_tokens 4 --compress_search_space True --optimal_class goldfish --max_n_calls 40000 --n_init_pts 280 --bsz 28
    # CUDA_VISIBLE_DEVICES=5 python3 optimize.py --n_tokens 4 --optimal_class goldfish --max_n_calls 40000 --n_init_pts 280 --bsz 28
    # CUDA_VISIBLE_DEVICES=6 python3 optimize.py --n_tokens 5 --compress_search_space True --optimal_class goldfish --max_n_calls 40000 --n_init_pts 280 --bsz 28
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --n_tokens 5 --optimal_class goldfish --max_n_calls 40000 --n_init_pts 280 --bsz 28


    # LOCUST (JAKE NEW MACHINE): ... ??? 
    #   tmux attach -t adv0, adv1, adv2, adv3, ..., adv7
    #   dockerd-rootless-setuptool.sh install
    #   systemctl --user start docker
    #   docker run -v /home1/n/nmaus/adversarial_img_bo/:/workspace/ --gpus all -it nmaus/advenv
    # CUDA_VISIBLE_DEVICES=0 python3 optimize.py --n_tokens 2 --compress_search_space True --optimal_class pug --max_n_calls 40000 --n_init_pts 2800 --bsz 28
    # CUDA_VISIBLE_DEVICES=1 python3 optimize.py --n_tokens 2 --optimal_class pug --max_n_calls 40000 --n_init_pts 2800 --bsz 28
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --n_tokens 3 --compress_search_space True --optimal_class pug --max_n_calls 40000 --n_init_pts 2800 --bsz 28
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --n_tokens 4 --optimal_class pug --max_n_calls 40000 --n_init_pts 2800 --bsz 28
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --n_tokens 4 --compress_search_space True --optimal_class pug --max_n_calls 40000 --n_init_pts 2800 --bsz 28
    # CUDA_VISIBLE_DEVICES=5 python3 optimize.py --n_tokens 4 --optimal_class pug --max_n_calls 40000 --n_init_pts 2800 --bsz 28
    # CUDA_VISIBLE_DEVICES=6 python3 optimize.py --n_tokens 5 --compress_search_space True --optimal_class pug --max_n_calls 40000 --n_init_pts 2800 --bsz 28
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --n_tokens 5 --optimal_class pug --max_n_calls 40000 --n_init_pts 2800 --bsz 28



    # Allegro (Osbert)
    #   tmux attach -t adv adv2, adv7  (now compression v2! )
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --n_tokens 4 --compress_search_space True --exclude_all_related_prompts True --optimal_class cat --prepend_task True --bsz 20
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --n_tokens 6 --compress_search_space True --exclude_all_related_prompts True --optimal_class cat --prepend_task True --bsz 20
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --n_tokens 6 --compress_search_space True --exclude_all_related_prompts True --optimal_class cat --bsz 20

    # gauss node 1,      conda activate adv_env   (no more exclusion at all, and down to 4 tokens instead of 6)
    #   tmux attach -t adv0, 1, 2, 3, ..., 8
    # CUDA_VISIBLE_DEVICES=0 python3 optimize.py --start_ix 0 --stop_ix 7 
    # CUDA_VISIBLE_DEVICES=1 python3 optimize.py --start_ix 7 --stop_ix 14 
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --start_ix 14 --stop_ix 21 
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --start_ix 21 --stop_ix 28 
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --start_ix 28  --stop_ix 35 
    # CUDA_VISIBLE_DEVICES=5 python3 optimize.py --start_ix 35 --stop_ix 42 
    # CUDA_VISIBLE_DEVICES=6 python3 optimize.py --start_ix 42 --stop_ix 49 
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --start_ix 49 --stop_ix 56 
    # CUDA_VISIBLE_DEVICES=8 python3 optimize.py --start_ix 56 --stop_ix 63  
    # gauss node 2, 
    #   tmux attach -t adv21 , adv22, adv23, adv24, adv29
    # CUDA_VISIBLE_DEVICES=1 python3 optimize.py --start_ix 63 --stop_ix 70 
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --start_ix 70 --stop_ix 77 
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --start_ix 77 --stop_ix 84 
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --start_ix 84 --stop_ix 92 
    # CUDA_VISIBLE_DEVICES=9 python3 optimize.py --start_ix 92 --stop_ix 200 --bsz 5
    # gauss node 3, (careful) 
    #   tmux attach -t adv1, adv2, adv6, adv7  (opt text! v2)
    # CUDA_VISIBLE_DEVICES=1 python3 optimize_text.py --n_tokens 5 --bsz 2 
    # CUDA_VISIBLE_DEVICES=2 python3 optimize_text.py --n_tokens 6 --bsz 10 
    # CUDA_VISIBLE_DEVICES=6 python3 optimize_text.py --n_tokens 4 --bsz 10 
    # CUDA_VISIBLE_DEVICES=7 python3 optimize_text.py --n_tokens 3 --bsz 10 

