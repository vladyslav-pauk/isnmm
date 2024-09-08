import wandb
import argparse


def init_wandb(api_key):
    wandb.login(key=api_key)


def download_best_model(run_id, project_name, model_filename="model.ckpt"):
    # Initialize wandb API
    api = wandb.Api()

    # Fetch the run
    run = api.run(f"{project_name}/{run_id}")

    # Download the model checkpoint
    run.file(model_filename).download(replace=True)

    print(f"Best model downloaded to {model_filename}")
    return model_filename


if __name__ == "__main__":
    # Parse arguments for API key
    # parser = argparse.ArgumentParser(description="Download the best model using Wandb API.")
    # parser.add_argument("--api_key", type=str, required=True, help="Wandb API Key")
    # args = parser.parse_args()

    # Initialize Weights & Biases with the API key
    import os
    os.environ["WANDB_API_KEY"] = "fcf64607eeb9e076d3cbfdfe0ea3532621753d78"
    wandb.login()

    run_id = "run-20240908_153536-seed_29-snr_20-lr_th_0.001-lr_ph_0.005"
    project_name = "vansca/vansca"
    model_filename = "best-model-epoch=00-val_r_squared=0.00.ckpt"

    checkpoint_path = download_best_model(run_id, project_name, model_filename)


# fixme: evaluate.py
# fixme: subspace metric
# fixme: check data model plots and inverses and expressiveness, choose seeds
# fixme: check neural network architecture and expressiveness, initialization
# fixme: adapt prism to new code
# fixme: run with only reconstruction term, check convergence, compare with prism, play with those
# fixme: clean and upload to github
# fixme: run schedule