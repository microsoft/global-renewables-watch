# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os
import pickle

import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument(
            "--experiment_name",
            type=str,
            default="dunnet_wind_2020q4_run_1_20_epochs",
            help="name of the experiment.",
        )

        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="1",
            help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
        )
        parser.add_argument(
            "--save_dir",
            type=str,
            required=True,
            help="latest models are saved here",
        )
        parser.add_argument(
            "--backup_dir",
            type=str,
            required=True,
            help="best models are saved here",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            required=True,
            help="directory where datasets are located",
        )
        parser.add_argument(
            "--init_type",
            type=str,
            default="none",
            help="which model to use [none | normal | xavier | xavier_uniform | kaiming | orthogonal]",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="fcn",
            help="which initialization method to use [unet | fcn | hrnet]",
        )
        parser.add_argument(
            "--phase", type=str, default="train", help="model phase [train | test]"
        )
        parser.add_argument(
            "--verbose", action="store_true", help="if set, print info while training"
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            default=False,
            help="if set, overwrite training dir",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="planetscope",
            help="model phase [bing | planetscope | dunnet]",
        )

        # input/output settings
        parser.add_argument(
            "--input_channels",
            type=int,
            default=3,
            help="Number of channel in the input images",
        )
        parser.add_argument(
            "--num_classes",
            type=int,
            default=2,
            help="Number of output segmentation classes per task",
        )
        parser.add_argument(
            "--num_workers", default=8, type=int, help="# workers for loading data"
        )

        # general model params
        parser.add_argument(
            "--first_layer_filters",
            type=int,
            default=8,
            help="Number of filters in the first UNet layer",
        )
        parser.add_argument(
            "--net_depth", type=int, default=4, help="Number of layers for the model"
        )

        parser.add_argument(
            "--checkpoint_file",
            type=str,
            default="none",
            help="Model to resume. If training from scratch use none",
        )

        self.initialized = True
        self.isTrain = False
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()

        if opt.checkpoint_file != "none":
            parser = self.update_options_from_file(parser, opt)

        # get the basic options
        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.save_dir, opt.experiment_name)
        if makedir:
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt")
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        print(file_name)
        with open(file_name + ".txt", "wt") as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ""
                default = self.parser.get_default(k)
                if v != default:
                    comment = "\t[default: %s]" % str(default)
                opt_file.write("{:>25}: {:<30}{}\n".format(str(k), str(v), comment))

        with open(file_name + ".pkl", "wb") as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + ".pkl", "rb"))
        return new_opt

    def parse(self, save=False):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, (
            "Batch size %d is wrong. It must be a multiple of # GPUs %d."
            % (opt.batchSize, len(opt.gpu_ids))
        )

        self.opt = opt
        return self.opt
