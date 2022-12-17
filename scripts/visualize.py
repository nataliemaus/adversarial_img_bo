
import torch
import sys 
sys.path.append("../")
from utils.adversarial_objective import AdversarialsObjective # best 
import argparse 
import os 
os.environ["WANDB_SILENT"] = "true" 


def load_tensor(wandb_run_name):
    return torch.load(f"../best_xs/{wandb_run_name}-best-x.pt")


def test_prompt(args, prompt, use_fixed_latents):
    objective = AdversarialsObjective(
        n_tokens=args.n_tokens,
        minimize=True, 
        batch_size=1,
        project_back=True,
        use_fixed_latents=use_fixed_latents,
        avg_over_N_latents=args.N_latents,
        allow_related_prompts=args.allow_related_prompts,
        seed=args.seed,
        prepend_to_text=args.prepend_to_text,
    ) 
    if not use_fixed_latents:
        for i in range(10):
            out_dict = objective.pipeline(
                input_type="prompt",
                input_value=prompt, 
                output_types=["image"],
                fixed_latents=objective.fixed_latents,
            )
            img = out_dict["image"][0]
            img.save(f"best_xs/{args.wandb_run_name}_test{i}_NOT_fixed_latents.png") 
    elif use_fixed_latents:
        out_dict = objective.pipeline(
            input_type="prompt",
            input_value=prompt, 
            output_types=["image"],
            fixed_latents=objective.fixed_latents,
        )
        img = out_dict["image"][0]
        img.save(f"best_xs/{args.wandb_run_name}_test1_fixed_latents.png") 
   

# v1 
# def visualize(args):
#     # assert args.n_tokens == 1 
#     objective = AdversarialsObjective(
#         n_tokens=args.n_tokens,
#         minimize=True, 
#         batch_size=1,
#         project_back=True,
#         use_fixed_latents=True,
#         visualize=True,
#         seed=args.seed,
#         allow_cat_prompts=args.allow_cat_prompts,
#         prepend_to_text=args.prepend_to_text,
#     ) 
#     assert objective.dim == args.n_tokens*768 
#     word_embedding = load_tensor(args.wandb_run_name).unsqueeze(0).to(torch.float16)
#     imgs, x, y = objective.query_oracle(word_embedding, return_img=True)
#     # imgs, y = objective.query_oracle(word_embedding, return_img=True)
#     img = imgs[0]
#     # outputs = objective.pipeline(
#     #     input_type = "word_embedding",
#     #     input_value = word_embedding,
#     #     output_types = ["image", "loss"],
#     #     fixed_latents=objective.fixed_latents
#     # )
#     # img = outputs["imgae"][0]
#     print("From prompt: ", x)
#     img.save(f"best_xs/{args.wandb_run_name}.png")

#     # test_prompt(args, prompt=x, use_fixed_latents=True) 
#     test_prompt(args, prompt=x, use_fixed_latents=False) 
#     import pdb 
#     pdb.set_trace() 


def visualizev2(args):
    objective = AdversarialsObjective(
        n_tokens=args.n_tokens,
        minimize=True, 
        batch_size=1,
        use_fixed_latents=False,
        project_back=True,
        avg_over_N_latents=args.N_latents,
        allow_related_prompts=args.allow_related_prompts,
        visualize=True,
        seed=args.seed,
        prepend_to_text=args.prepend_to_text,
    )
    if args.wandb_run_name == "random":
        best_x = torch.randn(1, objective.dim)
    else:
        best_x = load_tensor(args.wandb_run_name).unsqueeze(0).to(torch.float16)
    imgs, xs, y = objective.query_oracle(best_x, return_img=True)
    best_imgs = imgs[0] 
    for im_ix, img in enumerate(best_imgs):
        img.save(f"../best_xs/{args.wandb_run_name}_test{im_ix}.png")
    best_prompt = xs[0] 
    print("best prompt:", best_prompt)
    import pdb 
    pdb.set_trace() 
    # 'sphymk�'
    # torch.Size([1, 3, 768]) 
    # 'sphy','mk','�'

    # '<|endoftext|>meow<|endoftext|>'

def test_particular_prompt(args):
    objective = AdversarialsObjective(
        n_tokens=args.n_tokens,
        minimize=True, 
        batch_size=1,
        use_fixed_latents=False,
        project_back=True,
        avg_over_N_latents=args.N_latents,
        allow_related_prompts=args.allow_related_prompts,
        seed=args.seed,
        prepend_to_text=args.prepend_to_text,
    ) 
    if args.prepend_to_text:
        prompt = [args.prompt + " " + args.prepend_to_text + " <|endoftext|>" ]
    else:
        prompt = [args.prompt]
    for i in range(args.N_latents):
        out_dict = objective.pipeline(
            input_type="prompt",
            input_value=prompt, 
            output_types=["image"],
            fixed_latents=None,
        )
        img = out_dict["image"][0]
        img.save(f"../test_prompts/{prompt}_{args.n_tokens}tokens_test{i+1}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--wandb_run_name', default="firm-water-129" ) 
    parser.add_argument('--allow_related_prompts', type=bool, default=False ) 
    parser.add_argument('--N_latents', type=int, default=20 )   # N imagges
    parser.add_argument('--seed', type=int, default=2 )   

    parser.add_argument('--prompt', default="" )   # use exact prompt? 
    parser.add_argument('--prepend_to_text', default="a picture of a dog")
    parser.add_argument('--prepend_task', type=bool, default=False)
    parser.add_argument('--n_tokens', type=int, default=3 ) 
    args = parser.parse_args() 
    if not args.prepend_task: # if default task, prepend_to_text = ""
        args.prepend_to_text = ""
        
    if args.prompt: #if set prompt 
        test_particular_prompt(args)
    else:
        visualizev2(args) 
    

    # python3 visualize.py --prepend_task True --wandb_run_name random --N_latents 3
    # python3 visualize.py --prepend_task True --wandb_run_name random --N_latents 3 --prompt "pig pig pig"

    
    # conda activate lolbo_mols 
    # ['readkitty urban'] (giddy-moon-70) mostly cats! 
    # ['iz dal<|endoftext|><|endoftext|>siberian'] (rosy-meadow-71) 2/10 cats 

    # logical-monkey-132
    # PROMPT: asserounces standnocturnal shorthblackpool levi <|endoftext|>
    # TOKENIZED (8 tokens): asser,ounces,stand,nocturnal,shorth,blackpool,levi,<|endoftext|>

    # firm-water-129 
    # TOKENIZED: kitt,snow,anian,bob,response,<|endoftext|>

    # zany-forest-128
    # TOKENIZED: simon,adoptdont,batt,porta,sima,<|endoftext|>
    # adoptdont --> adopt dont shop ad campaign posters --> 1/3 are dogs 
