# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        # Options for training
        parser.add_argument(
            "--n_epochs", type=int, default=20, help="# of training epochs"
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            help="optimizer for training model [adam | sgd]",
        )
        parser.add_argument(
            "--beta1", type=float, default=0.9, help="momentum term of adam"
        )
        parser.add_argument(
            "--beta2", type=float, default=0.999, help="momentum term of adam"
        )

        parser.add_argument(
            "--batch_size", type=int, default=64, help="input batch size"
        )
        parser.add_argument(
            "--chip_size", type=int, default=256, help="input batch size"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0001, help="initial learning rate for adam"
        )
        parser.add_argument(
            "--loss",
            type=str,
            default="lcfcn",
            help="loss for training model [ce | lcfcn]",
        )
        parser.add_argument(
            "--scheduler_patience", type=int, default=5, help="lr scheduler patience"
        )
        parser.add_argument(
            "--segm_filter_size",
            type=int,
            default=7,
            help="size of filter around windmill center point for segmentation",
        )

        parser.add_argument(
            "--finetune_model",
            action="store_true",
            default=False,
            help="if set framework will finetune model. Also provide model to finetune",
        )
        parser.add_argument(
            "--model_to_finetune",
            type=str,
            default="path/to/checkpoint_best.pth.tar",
            help="if set framework will finetune model. Also provide model to finetune",
        )

        # CSRNet
        parser.add_argument(
            "--load_csrnet_weights",
            action="store_true",
            default=False,
            help="if set CSRNet load pretrained weights",
        )

        # HRNET
        parser.add_argument(
            "--cfg",
            help="experiment configure file for HRNet",
            default="options/hrnet.yaml",
            type=str,
        )

        self.isTrain = True
        return parser
