import json
import os
import yaml
from src.utils.utils import flatten_dict, unflatten_dict


def load_model_config(experiment, config_name):
    path = f'../src/model/config/{config_name}.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)


def load_sweep_config(experiment, config_name):
    path = f'../experiments/{experiment}/config/{config_name}.yaml'

    if os.path.exists(path):
        with open(path, 'r') as f:
            config = f.read()

    config = convert_yaml_to_json(config)
    return config


def convert_yaml_to_json(yaml_str):
    # Load YAML string as Python dictionary
    yaml_data = yaml.safe_load(yaml_str)

    # Function to transform the data structure as per the required JSON structure
    def transform_parameters(params):
        transformed = {}
        for key, value in params.items():
            # Convert scalar values into the required JSON structure
            if isinstance(value, list):
                transformed[key] = {"values": value}
            elif isinstance(value, dict):
                transformed[key] = {"values": [value]}
            else:
                transformed[key] = {"value": value}
        return transformed

    # Construct JSON-compatible dictionary
    json_data = {
        "method": yaml_data.get("method"),
        "metric": yaml_data.get("metric"),
        "parameters": transform_parameters(yaml_data.get("parameters", {}))
    }

    # Convert dictionary to JSON string
    return json_data


def load_data_config(experiment):
    path = f'../src/modules/data/synthetic.json'

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
