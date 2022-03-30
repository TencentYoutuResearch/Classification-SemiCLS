""" The Code is under Tencent Youtu Public Rule
This file provides utils for config management
"""
import logging

logger = logging.getLogger('root')

def overwrite_config(config, overwrite_string):
    '''
    other args to overwrite the config, keys are split by space
    and args split by |, the last element of a arg should be in
    [int float or string] for example

    seed 1|train trainer T 1
    '''
    args = overwrite_string.split("|")
    for arg in args:
        keys_value = arg.split(" ")
        value = keys_value[-1]

        for convert_type in [int, float, str]:
            try:
                value = convert_type(value)
                break
            except ValueError:
                continue

        if not isinstance(value, (float, int, str)):
            raise ValueError("Value {} should in int float str but \
                got {}".format(value, type(value)))

        keys = keys_value[:-1]
        logger.info("Overwriting {} to {}".format(keys, value))

        temp_cfg = config
        key_len = len(keys)
        for idx in range(key_len):
            if idx == key_len - 1:
                temp_cfg[keys[idx]] = value
            else:
                temp_cfg = temp_cfg[keys[idx]]
    return config
