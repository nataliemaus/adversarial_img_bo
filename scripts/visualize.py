
import torch
from utils.adversarial_objective import AdversarialsObjective # best 
import argparse 
import os 
os.environ["WANDB_SILENT"] = "true" 


def load_tensor(wandb_run_name):
    return torch.load(f"best_xs/{wandb_run_name}-best-x.pt")


def test_prompt(args, prompt, use_fixed_latents):
    objective = AdversarialsObjective(
        n_tokens=args.n_tokens,
        minimize=True, 
        batch_size=1,
        project_back=True,
        use_fixed_latents=use_fixed_latents,
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
   

def visualize(args):
    # assert args.n_tokens == 1 
    objective = AdversarialsObjective(
        n_tokens=args.n_tokens,
        minimize=True, 
        batch_size=1,
        project_back=True,
        use_fixed_latents=True,
    ) 
    assert objective.dim == args.n_tokens*768 
    word_embedding = load_tensor(args.wandb_run_name).unsqueeze(0).to(torch.float16)
    imgs, x, y = objective.query_oracle(word_embedding, return_img=True)
    # imgs, y = objective.query_oracle(word_embedding, return_img=True)
    img = imgs[0]
    # outputs = objective.pipeline(
    #     input_type = "word_embedding",
    #     input_value = word_embedding,
    #     output_types = ["image", "loss"],
    #     fixed_latents=objective.fixed_latents
    # )
    # img = outputs["imgae"][0]
    print("From prompt: ", x)
    img.save(f"best_xs/{args.wandb_run_name}.png")

    # test_prompt(args, prompt=x, use_fixed_latents=True) 
    test_prompt(args, prompt=x, use_fixed_latents=False) 
    import pdb 
    pdb.set_trace() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--wandb_run_name', default="giddy-moon-70" ) 
    parser.add_argument('--n_tokens', type=int, default=3 )     
    args = parser.parse_args() 
    visualize(args) 
    # activate lolbo_mols
    # ['readkitty urban'] (giddy-moon-70) mostly cats! 
    # ['iz dal<|endoftext|><|endoftext|>siberian'] (rosy-meadow-71) 2/10 cats 
