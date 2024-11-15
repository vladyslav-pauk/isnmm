import os
import json
import argparse
import hashlib
import yaml


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

def font_style():
    font = {
        'family': 'serif',
        # 'color': 'black',
        # 'weight': 'normal',
        'size': 22,
    }
    return font

def format_string(s):
    s = s.replace('_', ' ').title()
    abbr = ['Snr', 'Mse', 'Sam', 'Vasca', 'Nisca', 'Snae', 'Cnae']
    for a in abbr:
        if a in s:
            s = s.replace(a, a.upper())
    if 'Db' in s:
        s = s.replace('Db', 'dB')
    return s

# log_format = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(
#     level=logging.INFO,
#     format=log_format,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)