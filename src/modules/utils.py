def dict_to_str(d):
    return '_'.join([f'{value}' for key, value in d.items() if value is not None])
