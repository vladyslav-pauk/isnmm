import os
import json
import argparse
import hashlib


def sweep_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default='train_schedule',
        help='Experiment name (e.g., simplex_recovery)'
    )
    parser.add_argument(
        '--sweep',
        type=str,
        default='sweep',
        help='Sweep name (e.g., sweep)'
    )
    # parser.add_argument(
    #     '--data',
    #     nargs='+', default='lmm',
    #     help='List of datasets separated by space (e.g., lmm)'
    # )
    # parser.add_argument(
    #     '--models',
    #     nargs='+', default='vasca',
    #     help='List of models separated by space (e.g., vasca)'
    # )
    args = parser.parse_args()
    return args


def load_model_config(experiment, config_name):
    path = f'../experiments/{experiment}/config/model/{config_name}.json'

    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)


def load_sweep_config(experiment, config_name):
    path = f'../experiments/{experiment}/config/sweep/{config_name}.json'

    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)


def load_data_config(experiment):
    path = f'../experiments/{experiment}/config/data.json'

    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)


def update_hyperparameters(config, kwargs, show_log=True):
    unflattened_kwargs = unflatten_dict(kwargs)

    def print_flattened_dict(parent_key, flattened_dict):
        for flat_key, flat_value in flattened_dict.items():
            print(f"\t{parent_key}.{flat_key} = {flat_value}")

    for key, value in unflattened_kwargs.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
                flattened_value_dict = flatten_dict(value)
                if show_log:
                    print_flattened_dict(key, flattened_value_dict)
            else:
                config[key] = value
                if show_log:
                    print(f"\t{key} = {value}")

    return config


def get_parameter_combinations(config, prefix="", sep="_"):
    params = {}

    for key, value in config.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key

        if isinstance(value, dict):
            nested_params = get_parameter_combinations(value, new_key)
            params.update(nested_params)
        elif isinstance(value, list):
            params[new_key] = value

    return params


def hash_name(kwargs):
    if kwargs:
        kwargs_str = str(sorted(kwargs.items()))
        run_name = hashlib.md5(kwargs_str.encode()).hexdigest()
    else:
        run_name = None
    return run_name
    # if any(value is not None for value in kwargs.values()):
    #     run_name = "-".join([f"{key}_{value}" for key, value in kwargs.items() if value is not None])
    # else:
    #     run_name = None


def unflatten_dict(d, sep='.'):
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_ref = result
        for part in parts[:-1]:
            if part not in d_ref:
                d_ref[part] = {}
            d_ref = d_ref[part]
        d_ref[parts[-1]] = value
    return result


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

# log_format = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(
#     level=logging.INFO,
#     format=log_format,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)