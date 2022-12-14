
import torch
import gpytorch
import numpy as np 
import sys 
import copy 
sys.path.append("../") 
from gpytorch.mlls import PredictiveLogLikelihood 
from utils.bo_utils.trust_region import (
    TrustRegionState, 
    generate_batch, 
    update_state
)
from utils.bo_utils.ppgpr import (
    GPModelDKL,
    GPModel_Additive_Kernel,
    GPModelDKL_Additive_Kernel,
    SpecializedAdditiveGP,
)
from torch.utils.data import (
    TensorDataset, 
    DataLoader
)
from data.read_hierarchial_imagenet import load_imagenet_hierarcy_dicts
from utils.adversarial_objective import AdversarialsObjective  
import argparse 
import wandb 
import math 
import os 
os.environ["WANDB_SILENT"] = "true" 
import random 
import pandas as pd 
from utils.imagenet_classes import load_valid_imagenet_classes


class RunTurbo():
    def __init__(self, args):
        self.args = args 

    def initialize_global_surrogate_model(self, init_points, hidden_dims):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        if self.args.additive_gp:
            # good for single_number_per_token!!
            if self.args.single_number_per_token:
                if self.args.hidden_dims is None:
                    model = GPModel_Additive_Kernel(
                        inducing_points=init_points.cuda(), 
                        likelihood=likelihood,
                    )
                else:
                    model = GPModelDKL_Additive_Kernel(
                        inducing_points=init_points.cuda(), 
                        likelihood=likelihood,
                        hidden_dims=self.args.hidden_dims,
                    )
            else:
                # You'd have a sum of num_tokens kernels, 
                #   the first has active dims 0 through 767, 
                #   the second has active dims 768 through ...
                assert not self.args.compress_search_space 
                model = SpecializedAdditiveGP(
                    inducing_points=init_points.cuda(), 
                    likelihood=likelihood,
                    hidden_dims=self.args.hidden_dims,
                    num_tokens=self.args.n_tokens,
                )
        else:
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
        self.tracker = wandb.init(
            entity=args_dict['wandb_entity'], 
            project=args_dict['wandb_project_name'],
            config=args_dict, 
        ) 
        print('running', wandb.run.name) 

    def set_seed(self):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        seed = self.args.seed  
        if seed is not None:
            torch.manual_seed(seed) 
            random.seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(seed)

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


    def get_init_prompts(self):
        related_vocab = self.args.objective.related_vocab
        all_vocab = list(self.args.objective.vocab.keys()) 
        random.shuffle(all_vocab)
        starter_vocab = all_vocab[0:100]
        tmp = [] 
        for vocab_word in starter_vocab:
            if not vocab_word in related_vocab:
                tmp.append(vocab_word)
        starter_vocab = tmp 
        prompts = [] 
        iters = math.ceil(self.args.n_init_pts/self.args.n_init_per_prompt) 
        for i in range(iters):
            prompt = ""
            for j in range(self.args.n_tokens): # N
                if j > 0: 
                    prompt += " "
                # if i == 0:
                #     prompt += self.args.objective.optimal_class 
                # else:
                prompt += random.choice(starter_vocab)
            prompts.append(prompt)

        return prompts 
    

    def get_baseline_prompts(self):
        prompts = [] # 5 example baseline prompts 
        obj_cls = self.args.objective.optimal_class 
        # "CLS CLS CLS CLS" 
        prompt1 = obj_cls 
        for i in range(self.args.n_tokens - 1):
            prompt1 +=  f" {obj_cls }" 
        prompts.append(prompt1) 

        # "CLS end end end"
        prompt2 = obj_cls 
        for _ in range(self.args.n_tokens - 1):
            prompt2 += " <|endoftext|>"
        prompts.append(prompt2)

        # # "a picture of a CLS" 
        if self.args.n_tokens == 2:
            prompts.append(f"a {obj_cls}")
        elif self.args.n_tokens == 3:
            prompts.append(f"picture of {obj_cls}")
        elif self.args.n_tokens == 4:
            prompts.append(f"picture of a {obj_cls}")
        elif self.args.n_tokens == 5:
            prompts.append(f"a picture of a {obj_cls}")
        elif self.args.n_tokens > 5:
            prompt3 = f"a picture of a {obj_cls}"
            for _ in range(self.args.n_tokens - 5):
                prompt3 += " <|endoftext|>"
            prompts.append(prompt3)
    
        return prompts 

    def get_init_data(self ):
        # get scores for baseline_prompts 
        self.log_baseline_prompts() 
        # then get initialization prompts + scores ... 
        prompts = self.get_init_prompts()
        YS = [] 
        XS = [] 
        PS = []
        most_probable_clss = []
        prcnt_correct_clss = []
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
                most_probable_clss = most_probable_clss + self.args.objective.most_probable_classes
                prcnt_correct_clss = prcnt_correct_clss + self.args.objective.prcnts_correct_class
        Y = torch.cat(YS).detach().cpu() 
        Y = Y.unsqueeze(-1)  
        XS = torch.cat(XS).float().detach().cpu() 
        self.args.X = XS
        self.args.Y = Y 
        self.args.P = PS 
        self.args.most_probable_clss = most_probable_clss 
        self.args.prcnt_correct_clss = prcnt_correct_clss 

    def log_baseline_prompts(self):
        baseline_prompts = self.get_baseline_prompts() 
        while (len(baseline_prompts) % self.args.bsz) != 0:
            baseline_prompts.append(baseline_prompts[0])
        n_batches = int(len(baseline_prompts) / self.args.bsz )
        baseline_scores = []
        out_baseline_prompts = [] 
        baseline_most_probable_clss = []
        baseline_prcnt_correct_clss = []
        for i in range(n_batches): 
            prompt_batch = baseline_prompts[i*self.args.bsz:(i+1)*self.args.bsz] 
            X = self.args.objective.get_init_word_embeddings(prompt_batch) 
            X = X.detach().cpu() 
            X = X.reshape(self.args.bsz, self.args.objective.dim ) 
            xs, ys = self.args.objective(X.to(torch.float16))
            baseline_scores.append(ys) 
            out_baseline_prompts = out_baseline_prompts + xs
            baseline_most_probable_clss = baseline_most_probable_clss + self.args.objective.most_probable_classes
            baseline_prcnt_correct_clss = baseline_prcnt_correct_clss + self.args.objective.prcnts_correct_class 
        baseline_scores = torch.cat(baseline_scores).detach().cpu() # self.best_baseline_score
        self.best_baseline_score = baseline_scores.max().item()
        best_score_idx = torch.argmax(baseline_scores).item() 
        self.tracker.log({
            "baseline_scores":baseline_scores.tolist(),
            "baseline_prompts":out_baseline_prompts,
            "baseline_most_probable_classes":baseline_most_probable_clss,
            "baseline_prcnt_latents_correct_class_most_probables":baseline_prcnt_correct_clss,
            "best_baseline_score":self.best_baseline_score,
            "best_baseline_prompt":out_baseline_prompts[best_score_idx],
            "best_baseline_most_probable_class":baseline_most_probable_clss[best_score_idx],
            "best_baseline_prcnt_latents_correct_class_most_probable":baseline_prcnt_correct_clss[best_score_idx],
        }) 

    def save_stuff(self ):
        X = self.args.X
        Y = self.args.Y
        P = self.args.P 
        C = self.args.most_probable_clss
        PRC = self.args.prcnt_correct_clss 
        best_x = X[Y.argmax(), :].squeeze().to(torch.float16)
        torch.save(best_x, f"../best_xs/{wandb.run.name}-best-x.pt") 
        if self.args.objective.project_back: 
            best_prompt = P[Y.argmax()] 
            self.tracker.log({"best_prompt":best_prompt}) 
        # most probable class (mode over latents)
        most_probable_class = C[Y.argmax()] 
        self.tracker.log({"most_probable_class":most_probable_class}) 
        # prcnt of latents where most probable class is correct (ie 3/5)
        prcnt_latents_correct_class_most_probable = PRC[Y.argmax()] 
        self.tracker.log({"prcnt_latents_correct_class_most_probable":prcnt_latents_correct_class_most_probable}) 

        save_path = f"../best_xs/{wandb.run.name}-all-data.csv"
        df = pd.DataFrame() 
        df['prompt'] = np.array(P)
        df['most_probable_class'] = np.array(C)
        df['prcnt_latents_correct_class_most_probable'] = np.array(PRC) 
        df["loss"] = Y.squeeze().detach().cpu().numpy() 
        df.to_csv(save_path, index=None)


    def init_args(self):
        if self.args.debug:
            self.args.n_init_pts = 8
            self.args.init_n_epochs = 2 
            self.args.bsz = 2
            self.args.max_n_calls = 200
            self.args.avg_over_N_latents = 2 
            self.args.n_init_per_prompt = 2
        if self.args.n_init_per_prompt is None:
            self.args.n_init_per_prompt = 10 
        if not self.args.prepend_task: 
            self.args.prepend_to_text = ""
        if (self.args.n_tokens > 8) and self.args.more_hdims: # best cats and cars so far have n_tokens = 4, 6, and 8
            self.args.hidden_dims = tuple_type("(1024,256,128,64)") 
        if self.args.compress_search_space:
            self.args.hidden_dims = tuple_type("(32,32,16)") 
        if self.args.single_number_per_token:
            self.args.lb = 0
            self.args.ub = 1 
            if self.args.additive_gp:
                self.args.hidden_dims = None 
            else:
                self.args.hidden_dims = (self.args.n_tokens, self.args.n_tokens)
        else:
            self.args.lb = None
            self.args.ub = None
        # flags for wandb recording 
        self.args.update_state_fix = True 
        self.args.update_state_fix2 = True 
        self.args.update_state_fix3 = True 
        self.args.record_most_probable_fix2 = True 
        self.args.flag_set_seed = True
        self.args.flag_fix_args_reset = True  
        self.args.flag_reset_gp_new_data = True # reset gp every 10 iters up to 1024 data points 
        if self.args.n_init_pts is None:
            self.args.n_init_pts = self.args.bsz * self.args.n_init_per_prompt
        assert self.args.n_init_pts % self.args.bsz == 0

    def call_oracle_and_update_next(self, x_next):
        prompts_next, y_next = self.args.objective(x_next.to(torch.float16))
        self.args.P = self.args.P + prompts_next
        self.args.most_probable_clss = self.args.most_probable_clss + self.args.objective.most_probable_classes
        self.args.prcnt_correct_clss = self.args.prcnt_correct_clss + self.args.objective.prcnts_correct_class
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
            optimal_class_level=self.args.optimal_class_level,# 1,
            optimmal_sub_classes=self.args.optimmal_sub_classes, # [],
            visualize=False,
            compress_search_space=self.args.compress_search_space,
            single_number_per_token=self.args.single_number_per_token,
            remove_synonyms=self.args.remove_synonyms,
            lb = self.args.lb,
            ub = self.args.ub,
        )

    def optimize(self):
        self.set_seed()
        self.init_args()  
        self.start_wandb() # initialized self.tracker
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
        tr = TrustRegionState(
            dim=self.args.objective.dim,
            failure_tolerance=self.args.failure_tolerance,
            success_tolerance=self.args.success_tolerance,
        )
        n_iters = 0
        while self.args.objective.num_calls < self.args.max_n_calls:
            self.tracker.log({
                'num_calls':self.args.objective.num_calls,
                'best_y':self.args.Y.max(),
                'best_x':self.args.X[self.args.Y.argmax(), :].squeeze().tolist(), 
                'tr_length':tr.length,
                'tr_success_counter':tr.success_counter,
                'tr_failure_counter':tr.failure_counter,
                'num_tr_restarts':num_tr_restarts,
            } ) 
            if self.args.Y.max().item() > prev_best: 
                prev_best = self.args.Y.max().item() 
                # save_stuff(args, X, Y, P, args.objective, tracker)
                self.save_stuff()
            if self.args.break_after_success and (prev_best > self.best_baseline_score): 
                # only give n_addtional_evals more calls 
                self.args.max_n_calls = min(self.args.objective.num_calls + self.args.n_addtional_evals, self.args.max_n_calls)
                self.args.break_after_success = False 
                self.tracker.log({"beat_best_baseline":True})
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
            tr = update_state(tr, y_next) 
            if tr.restart_triggered:
                num_tr_restarts += 1
                tr = TrustRegionState(
                    dim=self.args.objective.dim,
                    failure_tolerance=self.args.failure_tolerance,
                    success_tolerance=self.args.success_tolerance,
                )
                model = self.initialize_global_surrogate_model(self.args.X, hidden_dims=self.args.hidden_dims) 
                model = self.update_surr_model(
                    model=model,
                    learning_rte=self.args.lr,
                    train_z=self.args.X,
                    train_y=self.args.Y,
                    n_epochs=self.args.init_n_epochs
                )
            # flag_reset_gp_new_data 
            elif (self.args.X.shape[0] < 1024) and (n_iters % 10 == 0): # reestart gp and update on all data 
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
            n_iters += 1
        self.tracker.finish() 


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
    parser.add_argument('--n_init_pts', type=int, default=None) 
    parser.add_argument('--prepend_to_text', default="a picture of a dog") 
    parser.add_argument('--break_after_success', type=bool, default=True )
    parser.add_argument('--success_value', default="beat_baseline" ) # type=int, default=-1)  
    # maybe later 
    parser.add_argument('--max_n_calls', type=int, default=5_000) 
    parser.add_argument('--n_addtional_evals', type=int, default=2_000) 
    parser.add_argument('--compression_version', type=int, default=2) # 2 == "laced-snow-14" 
    ## bsz ...  
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
    parser.add_argument('--single_number_per_token', type=bool, default=False )
    parser.add_argument('--failure_tolerance', type=int, default=32 )  
    parser.add_argument('--success_tolerance', type=int, default=10 )  
    parser.add_argument('--additive_gp', type=bool, default=False)  
    parser.add_argument('--optimal_class_level', type=int, default=1 )  
    parser.add_argument('--optimmal_sub_classes', type=list, default=[])  
    og_args = parser.parse_args() 

    if og_args.optimal_class != "all": # single class specified 
        runner = RunTurbo(og_args) 
        runner.optimize()
    else: 
        if og_args.optimal_class_level == 1:
            classes = load_valid_imagenet_classes()
        else:
            l2_to_l1, l3_to_l1 = load_imagenet_hierarcy_dicts(work_dir=og_args.work_dir) 
            d1 = l2_to_l1 
            if og_args.optimal_class_level == 3: d1 = l3_to_l1 
            classes = list(d1.keys())
        for clas in classes[og_args.start_ix:og_args.stop_ix]:
            args = copy.deepcopy(og_args)
            args.optimal_class = clas 
            if og_args.optimal_class_level > 1:
                args.optimmal_sub_classes = d1[clas]
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

    # LOCUST (JAKE NEW MACHINE): ... ??? 
    #   tmux attach -t adv0, adv1, adv2, adv3, ..., adv7
    #   dockerd-rootless-setuptool.sh install
    #   systemctl --user start docker
    #   docker run -v /home1/n/nmaus/adversarial_img_bo/:/workspace/ --gpus all -it nmaus/advenv
    # CUDA_VISIBLE_DEVICES=0 python3 optimize.py --optimal_class_level 2 --start_ix 0 --stop_ix 20 --bsz 28 
    # CUDA_VISIBLE_DEVICES=1 python3 optimize.py --optimal_class_level 2 --start_ix 20 --stop_ix 40 --bsz 28 
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --optimal_class_level 2 --start_ix 40 --stop_ix 60 --bsz 28 
    # ** CUDA_VISIBLE_DEVICES=3 python3 optimize.py --optimal_class_level 3 --start_ix 0 --stop_ix 100 --bsz 28 --exclude_some_related_prompts True  
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --optimal_class_level 2 --start_ix 60 --stop_ix 80 --bsz 28 
    # CUDA_VISIBLE_DEVICES=5 python3 optimize.py --optimal_class_level 2 --start_ix 80 --stop_ix 100 --bsz 25 
    # CUDA_VISIBLE_DEVICES=6 python3 optimize.py --optimal_class_level 2 --start_ix 100 --stop_ix 120 --bsz 28 
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --optimal_class_level 2 --start_ix 120 --stop_ix 145 --bsz 28 

    # ERIC MACHINE: ... ???   (ssh nmaus@deep-a6000x8-1.seas.upenn.edu )
    #   tmux attach -t adv0, adv1, adv2, adv3, ..., adv7
    #   dockerd-rootless-setuptool.sh install
    #   systemctl --user start docker
    #   docker run -v /home1/n/nmaus/adversarial_img_bo/:/workspace/ --gpus all -it nmaus/advenv
    # XXX LEAVE OPEN FOR TAI (CUDA_VISIBLE_DEVICES=0)
    # XXX LEAVE OPEN FOR TAI (CUDA_VISIBLE_DEVICES=1)
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --start_ix 360 --stop_ix 380 --bsz 28
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --start_ix 380 --stop_ix 400 --bsz 28
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --start_ix 400 --stop_ix 420 --bsz 28
    # CUDA_VISIBLE_DEVICES=5 python3 optimize.py --start_ix 420 --stop_ix 440 --bsz 28
    # XXX CUDA_VISIBLE_DEVICES=6 python3 optimize.py --start_ix 0 --stop_ix 200 --bsz 28 --seed 2
    # XXX CUDA_VISIBLE_DEVICES=7 python3 optimize.py --start_ix 200 --stop_ix 435 --bsz 28 --seed 2
    # CUDA_VISIBLE_DEVICES=6 python3 optimize.py --optimal_class_level 3 --start_ix 0 --stop_ix 13 --bsz 28 
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --optimal_class_level 3 --start_ix 13 --stop_ix 30 --bsz 28 

    # Allegro (Osbert)
    #   tmux attach -t adv5 adv6, adv7  (now compression + tr better) 
    # CUDA_VISIBLE_DEVICES=5 python3 optimize.py --start_ix 300 --stop_ix 320 --bsz 28
    # CUDA_VISIBLE_DEVICES=6 python3 optimize.py --start_ix 320 --stop_ix 340 --bsz 28
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --start_ix 340 --stop_ix 360 --bsz 28

    # gauss node 1,      conda activate adv_env   (no more exclusion at all, and down to 4 tokens instead of 6)
    #   tmux attach -t adv0, 1, 2, 3, ..., 8
    # CUDA_VISIBLE_DEVICES=0 python3 optimize.py --start_ix 0 --stop_ix 20 
    # CUDA_VISIBLE_DEVICES=1 python3 optimize.py --start_ix 20 --stop_ix 40 
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --start_ix 40 --stop_ix 60
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --start_ix 60 --stop_ix 80 
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --start_ix 80  --stop_ix 100 
    # CUDA_VISIBLE_DEVICES=5 python3 optimize.py --start_ix 100 --stop_ix 120 
    # CUDA_VISIBLE_DEVICES=6 python3 optimize.py --start_ix 120 --stop_ix 140 
    # CUDA_VISIBLE_DEVICES=7 python3 optimize.py --start_ix 140 --stop_ix 160 
    # CUDA_VISIBLE_DEVICES=8 python3 optimize.py --start_ix 160 --stop_ix 180 
    # gauss node 2, 
    #   tmux attach -t adv21 , adv22, adv23, adv24, adv29
    # CUDA_VISIBLE_DEVICES=1 python3 optimize.py --start_ix 180 --stop_ix 200 --bsz 10
    # CUDA_VISIBLE_DEVICES=2 python3 optimize.py --start_ix 200 --stop_ix 220 --bsz 8
    # CUDA_VISIBLE_DEVICES=3 python3 optimize.py --start_ix 220 --stop_ix 240 
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --start_ix 240 --stop_ix 260 
    # CUDA_VISIBLE_DEVICES=9 python3 optimize.py --start_ix 260 --stop_ix 280 --bsz 8
    # gauss node 3, (careful) 
    #   tmux attach -t adv4 
    # CUDA_VISIBLE_DEVICES=4 python3 optimize.py --start_ix 280 --stop_ix 300 --bsz 8

    # 433 total valid one word classes 