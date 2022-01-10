# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file.

import copy
from easydict import EasyDict as edict
import os
import yaml


def read_yaml(filename):
    """Load yaml file as a dictionary item

    Args:
        filename (str): .yaml file path

    Returns:
        cfg (dict): configuration
    """

    if filename is not None:
        with open(filename, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        return {}


def nested_dict(opts, value):
    """create a nested dictionary given a list of keys and a value
    """

    if len(opts) > 1:
        return {opts[0]: nested_dict(opts[1:], value)}
    elif len(opts) == 1:
        return {opts[0]: value}
    else:
        ValueError


class ConfigLoader():    
    def update_dict(self, dict1, dict2):
        """update dict1 according to dict2
        Args:
            dict1 (dict): reference dictionary
            dict2 (dict): new dictionary
        return
            dict1 (dict): updated reference dictionary
        """
        for item in dict2:
            if dict1.get(item, -1) != -1:
                if isinstance(dict1[item], dict):
                    dict1[item] = self.update_dict(dict1[item], dict2[item])
                else:
                    if isinstance(dict1[item], bool) and isinstance(dict2[item], str):
                        dict2[item] = dict2[item].lower() in ["true"]
                    dict1[item] = type(dict1[item])(dict2[item])
            else:
                dict1[item] = dict2[item]
        return dict1

    def merge_cfg(self, cfg_files):
        """merge default configuration and custom configuration

        Args:
            cfg_files (str): configuration file paths [default, custom]
        Returns:
            cfg (edict): merged EasyDict
        """
        cfg = {}
        for f in cfg_files:
            if f is not None:
                if isinstance(f, str):
                    assert os.path.isfile(f), "config file not exists"
                    cfg = self.update_dict(cfg, read_yaml(f))
                if isinstance(f, dict):
                    cfg = self.update_dict(cfg, f)
        return edict(cfg)

    def merge_args(self, cfg, opts):
        """merge config with arguments from command line
        
        Args:
            cfg (dict): old config
            opts (str): new arguments from command line
        Returns:
            cfg (dict): updated config
        """
        cfg = copy.deepcopy(cfg)
        if opts is None:
            return cfg
        for opt in opts:
            keys, value = opt.split(":")
            keys = keys.split(".")
            opt_dict = nested_dict(keys, value)
            self.merge_cfg([cfg, opt_dict])
        return cfg

    def write_cfg(self, default, merge, f, level_cnt=0):
        """write configuration to file
        Args:
            default (dict): default configuration dictionary
            merge (dict): merged configuration dictionary
            file (TextIOWrapper)
        """
        offset_len = 100
        for item in merge:
            if isinstance(merge[item], dict):
                # go deeper for dict item
                line = "  "*level_cnt + item + ": "
                offset = offset_len - len(line)
                line += " "*offset + " # | "
                
                # check if default has this config
                if default.get(item, -1) == -1:
                    default[item] = {}
                    line += " --NEW-- "
                f.writelines(line + "\n")
                self.write_cfg(default[item], merge[item], f, level_cnt+1)
            else:
                # write current config
                line = "  " * level_cnt + item + ": "
                if merge[item] is not None:
                    line += str(merge[item])
                else:
                    line += " "
                
                offset = offset_len - len(line)
                line += " "*offset + " # | "
                f.writelines(line)

                # write default if default is different from current
                if default.get(item, -1) != -1:
                    line = " "
                    if merge[item] != default[item]:
                        line = str(default[item])
                    f.writelines(line)
                else:
                    line = " --NEW-- "
                    f.writelines(line)
                f.writelines("\n")

    def save_cfg(self, cfg_files, file_path):
        """Save configuration file
        Args:
            cfg_files (str): configuration file paths [default, custom]
            file_path (str): path of text file for writing the configurations
        """
        # read configurations
        default = read_yaml(cfg_files[0])

        # create file to be written
        f = open(file_path, 'w')

        # write header line
        line = "# " + "-"*20 + " Setup " + "-"*74
        line += "|" + "-"*10 + " Default " + "-"*20 + "\n"
        f.writelines(line)

        # merge configurations
        merged = self.merge_cfg(cfg_files)

        # write configurations
        self.write_cfg(default, merged, f)
        f.close()