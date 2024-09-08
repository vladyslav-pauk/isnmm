from train import run_training

lr_th_values = [0.001, 0.01]
lr_ph_values = [0.01, 0.005]
snr_values = [10, 20, 30]
seed_values = [29, 0, 42]

model_name = 'vansca'

for seed in seed_values:
    for snr in snr_values:
        for lr_th in lr_th_values:
            for lr_ph in lr_ph_values:
                print(f"Running training with seed={seed}, snr={snr}, lr_th={lr_th}, lr_ph={lr_ph}")
                run_training(model_name, lr_th=lr_th, lr_ph=lr_ph, snr=snr, seed=seed)

                # todo: Log or handle results
                # print(f"Training complete with true_A: {true_A}, est_A: {est_A}")

# todo: https://pytorch-lightning.readthedocs.io/en/0.9.0/hyperparameters.html