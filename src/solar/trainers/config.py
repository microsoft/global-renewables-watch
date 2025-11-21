# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import timm
import torch
from loguru import logger
from pydantic import BaseModel, root_validator, validator

GPUS_AVAILABLE = torch.cuda.device_count()
BACKBONE_OPTIONS = timm.list_models(pretrained=True)


##################################################
### Define user choices to validate yaml input ###
##################################################


class ModelEnum(str, Enum):
    """Models supported by train.py."""

    unet = "unet"
    deeplab = "deeplabv3+"
    fcn = "fcn"


class WeightEnum(str, Enum):
    """Weights supported by train.py."""

    imagenet = "imagenet"
    random = "random"
    ssl = "ssl"


class OptimizerEnum(str, Enum):
    """Optimizers supported by train.py."""

    adam = "adam"
    rmsprop = "rmsprop"
    sgd = "sgd"
    adamw = "adamw"


class LossEnum(str, Enum):
    """Losses supported by train.py."""

    ce = "ce"
    jaccard = "jaccard"
    focal = "focal"
    wbce = "wbce"
    dicewbce = "dicewbce"


class SchedulerEnum(str, Enum):
    """Schedulers supported by train.py."""

    cosine = "cosine"
    plateau = "plateau"


######################################
### Set up Configs for each module ###
######################################


class SegmentationBaseModel(BaseModel):
    """Set defaults for all models that inherit from the pydantic base model."""

    class Config:
        extra = "forbid"
        use_enum_values = True
        validate_assignment = True


######################################
### Validate final training config ###
######################################


class TrainerConfig(SegmentationBaseModel):
    """Validate input from yaml and/or argparse before passing to train.py."""

    # model params
    model_name: ModelEnum = ModelEnum.unet
    backbone_name: str = "resnet18"
    weight_init: WeightEnum = WeightEnum.imagenet

    # optimizer params
    optimizer: OptimizerEnum = OptimizerEnum.adamw
    lr: float = 0.001
    weight_decay: float = 0.01
    patience: int = 6
    scheduler: SchedulerEnum = SchedulerEnum.plateau

    # loss params
    loss: LossEnum = LossEnum.ce
    wbce_weight: Optional[float] = 0.7

    # data module params
    patch_size: int = 512
    batch_size: int = 24
    num_workers: int = 6
    batches_per_epoch: int = 256
    color_jitter_augmentation: bool = True
    no_sharpness_augmentation: bool = False

    # trainer params
    gpu_ids: List[int] = [0]
    seed: int = 0
    max_epochs: int = 30
    log_dir: str = "logs/"
    output_dir: str = "model_runs/"
    overwrite: bool = True

    # input data params
    experiment_short_name = "example_experiment"
    dir_parent: Optional[str] = "/home/tammyglazer/ssdshared/solar/"
    subdirs: Optional[List[str]] = ["test1/", "test2/"]

    # generated during validation if not explicitly passed in
    input_dirs: Optional[Union[List[str], str]] = None
    experiment_name: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("gpu_ids", always=True)
    def validate_gpus(cls, gpu_ids):
        for gpu_id in gpu_ids:
            if gpu_id >= GPUS_AVAILABLE:
                raise ValueError(
                    f"Found only {GPUS_AVAILABLE} GPU(s). Cannot use {gpu_id}."
                )
        logger.info(f"Using the following GPU(s): {gpu_ids}.")
        return gpu_ids

    @root_validator(skip_on_failure=True)
    def validate_input_dirs(cls, values):
        logger.info("Validating input directories.")

        # if full input_dirs are provided, check that they exist
        if values["input_dirs"]:
            for fp in values["input_dirs"]:
                if not Path(fp).exists():
                    raise ValueError(f"{fp} does not exist.")
            values["subdirs"] = None
            values["dir_parent"] = None
            return values
        else:
            parent_dir = Path(values["dir_parent"])
            fps = [parent_dir / x for x in values["subdirs"]]
            for fp in fps:
                if not fp.exists():
                    raise ValueError(f"{fp} does not exist.")

            # set input_dirs if not passed in CLI
            str_fps = [str(fp) for fp in fps]
            values["input_dirs"] = str_fps
            return values

    @root_validator(skip_on_failure=True)
    def validate_experiment_name(cls, values):
        # if experiment_name is provided, use it
        if values["experiment_name"]:
            return values
        else:
            logger.info("Constructing experiment name.")
            short_name = values["experiment_short_name"]
            model_name = values["model_name"]
            backbone_name = values["backbone_name"]
            weight_init = values["weight_init"]
            lr = values["lr"]
            weight_decay = values["weight_decay"]
            patience = values["patience"]
            batch_size = values["batch_size"]
            loss = values["loss"]
            seed = values["seed"]
            if values["color_jitter_augmentation"]:
                aug_color_jitter = "--aug_color_jitter"
            else:
                aug_color_jitter = ""
            if values["loss"] == "wbce":
                ww = values["wbce_weight"]
                wbce_weight = f"--wbce-weight_{ww}"
            else:
                wbce_weight = ""

            experiment_name = (
                f"{short_name}--{model_name}--{backbone_name}--{weight_init}--lr_{lr}"
                + f"--wd_{weight_decay}--lr-patience_{patience}--bs_{batch_size}--loss_{loss}--seed_{seed}"
                + f"{aug_color_jitter}"
                + f"{wbce_weight}"
            )
            values["experiment_name"] = experiment_name
            return values

    @root_validator(skip_on_failure=True)
    def validate_output_dir(cls, values):
        logger.info("Validating log directory.")
        log_dir = values["log_dir"]
        output_dir = values["output_dir"]
        experiment = values["experiment_name"]

        output_log_dir = Path(log_dir)
        output_run_dir = Path(output_dir) / experiment

        if (output_log_dir.exists() or output_run_dir.exists()) and not values[
            "overwrite"
        ]:
            raise ValueError(
                "Output directories already exist and overwrite is not set to true."
            )
        output_log_dir.mkdir(parents=True, exist_ok=True)
        values["log_dir"] = str(output_log_dir)

        output_run_dir.mkdir(parents=True, exist_ok=True)
        values["output_dir"] = str(output_run_dir)
        return values
