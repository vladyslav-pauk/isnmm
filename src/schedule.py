import itertools
import wandb
import argparse

from train import train_model
from src.helpers.utils import load_experiment_config, get_parameter_combinations
from src.generate_data import initialize_data_model

# todo: fix and run schedule with 100 MC runs.
# todo: implement saving of multiple run results mean and vars and comparison of models
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default='train_schedule',
        help='Experiment name (e.g., simplex_recovery)'
    )
    parser.add_argument(
        '--data',
        nargs='+', default='lmm',
        help='List of datasets separated by space (e.g., lmm)'
    )
    parser.add_argument(
        '--models',
        nargs='+', default='vasca',
        help='List of models separated by space (e.g., vasca)'
    )
    args = parser.parse_args()

    experiment = args.experiment
    models = args.models
    datasets = args.data

    print(f"Starting experiment '{experiment}'")

    for dataset in datasets:
        for model in models:
            config = load_experiment_config(experiment, model)
            param_dict = get_parameter_combinations(config["schedule"]["parameters"])

            param_keys = list(param_dict.keys())
            param_values = list(param_dict.values())

            parameter_combinations = list(itertools.product(*param_values))

            for param_combination in parameter_combinations:
                kwargs = dict(zip(param_keys, param_combination))

                print(f"Generating dataset '{dataset}'")
                data_model = initialize_data_model(experiment, dataset, **kwargs)
                data_model.sample()
                data_model.save_data()

                print(f"Training '{model}' with parameters: \n\t{kwargs}")
                for iteration in range(config["schedule"]['repeats']):
                    model = train_model(experiment, dataset, model, **kwargs)
                    # load each model and model.test() to get the results, collect the results and plot: snr x-axis, seeds are giving variance of the shaded area

                wandb.finish()

# todo: name of schedule on wandb to separate scheduled runs
# todo: implement wandb sweep
# todo: run with configuration like train.py, argparse
# todo: group runs by same hyperparameter setting
# todo: fix order of training, seed last
# todo: make it display link to wandb even with wandb.finish() command
# todo: Log or handle (print) results
# print(f"Training complete with true_A: {true_A}, est_A: {est_A}")
# todo: https://pytorch-lightning.readthedocs.io/en/0.9.0/hyperparameters.html


# import wandb
# import argparse
# from train import train_model
# from src.helpers.utils import load_experiment_config
# from src.modules.data_module import DataModule
#
#
# # Function to flatten the nested dictionary
# def flatten_dict(d, parent_key='', sep='.'):
#     """
#     Recursively flattens a nested dictionary. The keys of the nested dictionary
#     are joined with the `sep` string.
#
#     Example:
#     {'lr': {'th': [0.001, 0.01], 'ph': [0.01, 0.005]}}
#     -> {'lr.th': [0.001, 0.01], 'lr.ph': [0.01, 0.005]}
#     """
#     items = []
#     for k, v in d.items():
#         new_key = parent_key + sep + k if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)
#
#
# # Function to create the sweep configuration dynamically from argparse
# def create_sweep_config(experiment, datasets, models, schedule_config):
#     """
#     Create a Wandb sweep configuration dynamically based on argparse input.
#     """
#     # Flatten the parameters in the schedule config
#     flat_parameters = flatten_dict(schedule_config['parameters'])
#     flat_data = flatten_dict(schedule_config['data'])
#
#     # Define the sweep config using values from the flattened schedule config
#     sweep_config = {
#         'method': 'grid',  # Can also be 'random' or 'bayes'
#         'metric': {
#             'name': 'validation_loss',
#             'goal': 'minimize'
#         },
#         'parameters': {
#             'experiment': {
#                 'values': [experiment]
#             },
#             'dataset': {
#                 'values': datasets
#             },
#             'model': {
#                 'values': models
#             }
#         }
#     }
#
#     # Add flattened schedule parameters to the sweep config
#     for key, values in flat_parameters.items():
#         sweep_config['parameters'][key] = {'values': values}
#
#     # Add flattened data parameters to the sweep config
#     for key, values in flat_data.items():
#         sweep_config['parameters'][key] = {'values': values}
#
#     return sweep_config
#
#
# # Sweep function that defines the training loop
# def sweep_train():
#     # Initialize a new run with wandb
#     wandb.init()
#
#     # Get hyperparameters from wandb.config
#     config = wandb.config
#
#     # Load the experiment and data configurations
#     experiment = config.experiment
#     model_name = config.model
#     dataset = config.dataset
#
#     # Log some information in wandb run
#     wandb.run.tags = [experiment, dataset, model_name]
#
#     # Load the dataset config and initialize the DataModule
#     data_config = load_experiment_config(experiment, dataset)
#
#     # Load the model config
#     model_config = load_experiment_config(experiment, model_name)
#
#     # Combine the model config and data config into one config
#     combined_config = model_config
#     combined_config["data"] = data_config["data"]  # Ensure 'data' is properly passed
#
#     # Initialize the DataModule
#     datamodule = DataModule(data_config)
#
#     # Extract schedule-related configurations
#     schedule_params = {
#         key: config[key] for key in config.keys() if key.startswith("lr") or key.startswith("SNR")
#     }
#
#     # Get the parameter combinations based on the sweep
#     kwargs = {key: value for key, value in schedule_params.items()}
#
#     # Manually control repeats inside the training loop
#     num_repeats = 3  # Adjust the number of repeats here
#     for iteration in range(num_repeats):
#         print(f"Training '{model_name}' on dataset '{dataset}' with parameters: {kwargs}, iteration {iteration + 1}")
#         train_model(experiment, combined_config, datamodule, **kwargs)
#
#     wandb.finish()
#
#
# if __name__ == '__main__':
#     import os
#     os.environ["WANDB_API_KEY"] = "fcf64607eeb9e076d3cbfdfe0ea3532621753d78"
#     os.environ['WANDB_SILENT'] = 'true'
#     wandb.require("core")
#     # Argparse for command line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--experiment',
#         type=str,
#         default='train_schedule',
#         help='Experiment name (e.g., simplex_recovery)'
#     )
#     parser.add_argument(
#         '--data',
#         nargs='+', default=['lmm'],
#         help='List of datasets separated by space (e.g., lmm)'
#     )
#     parser.add_argument(
#         '--models',
#         nargs='+', default=['vasca'],
#         help='List of models separated by space (e.g., vasca)'
#     )
#     args = parser.parse_args()
#
#     experiment = args.experiment
#     models = args.models
#     datasets = args.data
#
#     print(f"Starting experiment '{experiment}' with datasets {datasets} and models {models}")
#
#     for dataset in datasets:
#         # Load the dataset configuration and initialize the DataModule
#         data_config = load_experiment_config(experiment, dataset)
#         datamodule = DataModule(data_config)
#
#         for model in models:
#             # Load model configuration
#             model_config = load_experiment_config(experiment, model)
#             schedule_config = model_config["schedule"]
#
#             # Create the sweep config based on the argparse input and schedule config
#             sweep_config = create_sweep_config(experiment, datasets, models, schedule_config)
#
#             # Create the sweep
#             sweep_id = wandb.sweep(sweep_config, project="simplex_recovery")
#
#             # Launch the sweep agent (run multiple experiments)
#             wandb.agent(sweep_id, function=sweep_train)