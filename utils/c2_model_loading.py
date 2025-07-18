#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Caffe2 to PyTorch checkpoint name converting utility."""

import re
import pickle
from collections import OrderedDict


def get_name_convert_func():
    """
    Get the function to convert Caffe2 layer names to PyTorch layer names.
    Returns:
        (func): function to convert parameter name from Caffe2 format to PyTorch
        format.
    """
    pairs = [
        # ------------------------------------------------------------
        # 'nonlocal_conv3_1_theta_w' -> 's3.pathway0_nonlocal3.conv_g.weight'
        [
            r"^nonlocal_conv([0-9]+)_([0-9]+)_(.*)",
            r"s\1.pathway0_nonlocal\2_\3",
        ],
        # 'theta' -> 'conv_theta'
        [r"^(.*)_nonlocal([0-9]+)_(theta)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'g' -> 'conv_g'
        [r"^(.*)_nonlocal([0-9]+)_(g)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'phi' -> 'conv_phi'
        [r"^(.*)_nonlocal([0-9]+)_(phi)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'out' -> 'conv_out'
        [r"^(.*)_nonlocal([0-9]+)_(out)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'nonlocal_conv4_5_bn_s' -> 's4.pathway0_nonlocal3.bn.weight'
        [r"^(.*)_nonlocal([0-9]+)_(bn)_(.*)", r"\1_nonlocal\2.\3.\4"],
        # ------------------------------------------------------------
        # 't_pool1_subsample_bn' -> 's1_fuse.conv_f2s.bn.running_mean'
        [r"^t_pool1_subsample_bn_(.*)", r"s1_fuse.bn.\1"],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [r"^t_pool1_subsample_(.*)", r"s1_fuse.conv_f2s.\1"],
        # 't_res4_5_branch2c_bn_subsample_bn_rm' -> 's4_fuse.conv_f2s.bias'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_bn_(.*)",
            r"s\1_fuse.bn.\3",
        ],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_(.*)",
            r"s\1_fuse.conv_f2s.\3",
        ],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway0_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^res_conv1_bn_(.*)", r"s1.pathway0_stem.bn.\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway0_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^res_conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway1_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^t_res_conv1_bn_(.*)", r"s1.pathway1_stem.bn.\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^t_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway1_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^t_res_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # ------------------------------------------------------------
        # pred_ -> head.projection.
        [r"pred_(.*)", r"head.projection.\1"],
        # '.bn_b' -> '.weight'
        [r"(.*)bn.b\Z", r"\1bn.bias"],
        # '.bn_s' -> '.weight'
        [r"(.*)bn.s\Z", r"\1bn.weight"],
        # '_bn_rm' -> '.running_mean'
        [r"(.*)bn.rm\Z", r"\1bn.running_mean"],
        # '_bn_riv' -> '.running_var'
        [r"(.*)bn.riv\Z", r"\1bn.running_var"],
        # '_b' -> '.bias'
        [r"(.*)[\._]b\Z", r"\1.bias"],
        # '_w' -> '.weight'
        [r"(.*)[\._]w\Z", r"\1.weight"],
    ]

    def convert_caffe2_name_to_pytorch(caffe2_layer_name):
        """
        Convert the caffe2_layer_name to pytorch format by apply the list of
        regular expressions.
        Args:
            caffe2_layer_name (str): caffe2 layer name.
        Returns:
            (str): pytorch layer name.
        """
        for source, dest in pairs:
            caffe2_layer_name = re.sub(source, dest, caffe2_layer_name)
        return caffe2_layer_name

    return convert_caffe2_name_to_pytorch


def convert_facebook_pretrained_model(ckpt_pth):
    with open(ckpt_pth, 'rb') as f:
        ckpt = pickle.load(f, encoding='latin1')
    state_dict = OrderedDict()
    name_convert_func = get_name_convert_func()
    temp_dict = {}
    for key in ckpt["blobs"].keys():
        converted_key = name_convert_func(key)
        if key == 'model_iter':
            continue
        else:
            temp_dict[converted_key] = ckpt["blobs"][key]
    sorted_keys = sorted(temp_dict.keys(), reverse=False)
    for key in sorted_keys:
        if "momentum" in key:
            continue
        converted_key = ""
        temp = ""
        if "pathway0_" in key:
            converted_key = "slow_"
        if "pathway1_" in key:
            converted_key = "fast_"
        if "_stem" in key:
            if ".bn" in key:
                temp = re.sub(
                    r"^s1.pathway\d_stem.bn(.*)",
                    r"bn1\1",
                    key
                )
            if ".conv" in key:
                temp = re.sub(
                    r"^s1.pathway\d_stem.conv(.*)",
                    r"conv1\1",
                    key
                )
            converted_key = converted_key + temp
            state_dict[converted_key] = temp_dict[key]
            continue

        if "_fuse" in key:
            converted_key = "lateral_"
            if "s1_" in key:
                temp = re.sub(
                    r"^s1_fuse(.*)",
                    r"p1\1",
                    key
                )
                converted_key = converted_key + temp
                state_dict[converted_key] = temp_dict[key]
                continue
            else:
                temp = re.sub(
                    r"^s(\d)_fuse(.*)",
                    r"res\1\2",
                    key
                )
                converted_key = converted_key + temp
                state_dict[converted_key] = temp_dict[key]
                continue

        if "branch1" in key:
            # downsample
            if "bn" not in key:
                temp = re.sub(
                    r"^s(\d).pathway\d_res(\d).branch1(.*)",
                    r"res\1.\2.downsample.0\3",
                    key
                )
                converted_key = converted_key + temp
                state_dict[converted_key] = temp_dict[key]
                continue
            else:
                temp = re.sub(
                    r"^s(\d).pathway\d_res(\d).branch1_bn\.(.*)",
                    r"res\1.\2.downsample.1.\3",
                    key
                )
                converted_key = converted_key + temp
                state_dict[converted_key] = temp_dict[key]
                continue

        if 'branch2' in key:
            if "bn" not in key:
                temp = re.sub(
                    r"^s(\d).pathway\d_res(\d).branch2.[a-z].weight",
                    r"res\1.\2.convZ.weight",
                    key
                )
                converted_key = converted_key + temp
                converted_key = converted_key.replace('Z', str(ord(key.split('.')[-2][0])-ord('a')+1))
                state_dict[converted_key] = temp_dict[key]
                continue
            else:
                temp = re.sub(
                    r"^s(\d).pathway\d_res(\d).branch2.[a-z]_bn(.*)",
                    r"res\1.\2.bnZ\3",
                    key
                )
                converted_key = converted_key + temp
                converted_key = converted_key.replace('Z', str(ord(key.split('.')[-2][0]) - ord('a') + 1))
                state_dict[converted_key] = temp_dict[key]
                continue

    return state_dict
