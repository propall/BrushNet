#!/usr/bin/env python
# coding=utf-8

import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import json
import cv2

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    BrushNetModel,
    DDPMScheduler,
    StableDiffusionBrushNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# Import your custom dataset class here
from custom_dataset import CustomBrushNetDataset, collate_fn

if is_wandb_available():
    import wandb

check_min_version("0.27.0.dev0")
logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="BrushNet training script for custom dataset.")
    
    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="data/ckpt/realisticVisionV60B1_v51VAE",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--brushnet_model_name_or_path",
        type=str,
        default="data/ckpt/brushnetX",
        help="Path to pretrained brushnet model or model identifier from huggingface.co/models."
        " If not specified brushnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    
    # **MODIFIED: Changed from train_data_dir to dataset_dir**
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/render_dataset",
        required=True,
        help=(
            "Path to the dataset directory containing 'images/', 'masks/', and 'captions.json'. "
            "This replaces the webdataset functionality with a simpler folder structure."
        ),
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/ckpt/brushnetX-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all images will be resized to this resolution",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    # Validation arguments  
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=["A beautiful floor"],
        nargs="+",
        help="A set of prompts evaluated every `--validation_steps`.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help="A set of paths to validation images.",
    )
    parser.add_argument(
        "--validation_mask",
        type=str,
        default=None,
        nargs="+",
        help="A set of paths to validation masks.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each validation prompt",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=250,
        help="Run validation every X steps.",
    )
    
    # Other arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--random_mask",
        action="store_true",
        help="Training BrushNet with random mask generation instead of provided masks",
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.1,
        help="Proportion of image prompts to be replaced with empty strings.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to. Supported platforms are "tensorboard", "wandb".',
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Validation
    if not os.path.exists(args.dataset_dir):
        raise ValueError(f"Dataset directory {args.dataset_dir} does not exist")
    
    required_paths = [
        os.path.join(args.dataset_dir, "images"),
        os.path.join(args.dataset_dir, "masks"), 
        os.path.join(args.dataset_dir, "captions.json")
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise ValueError(f"Required path {path} does not exist in dataset directory")

    return args


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """Import the appropriate text encoder class"""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def main(args):
    """Main training function"""
    
    # Setup logging and accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    
    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # Load or initialize BrushNet
    if args.brushnet_model_name_or_path:
        logger.info("Loading existing brushnet weights")
        brushnet = BrushNetModel.from_pretrained(args.brushnet_model_name_or_path)
    else:
        logger.info("Initializing brushnet weights from unet")
        brushnet = BrushNetModel.from_unet(unet)

    # Freeze base models, only train BrushNet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    brushnet.train()

    # **MODIFIED: Create custom dataset instead of webdataset**
    train_dataset = CustomBrushNetDataset(
        dataset_dir=args.dataset_dir,
        resolution=args.resolution,
        tokenizer=tokenizer,
        random_mask=args.random_mask
    )
    
    # **MODIFIED: Use standard PyTorch DataLoader instead of webdataset loader**
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Enable memory optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            brushnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        brushnet.enable_gradient_checkpointing()

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        brushnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with accelerator
    brushnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        brushnet, optimizer, train_dataloader, lr_scheduler
    )

    # Move models to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # **TRAINING LOOP - This is where the core objective function is applied**
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(brushnet):
                # **CORE DIFFUSION OBJECTIVE FUNCTION IMPLEMENTATION**
                
                # 1. Encode images to latent space using VAE
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2. Encode conditioning (masked) images to latent space  
                conditioning_latents = vae.encode(batch["conditioning_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                conditioning_latents = conditioning_latents * vae.config.scaling_factor

                # 3. Resize and concatenate mask information
                masks = torch.nn.functional.interpolate(
                    batch["masks"], 
                    size=(latents.shape[-2], latents.shape[-1])
                )
                conditioning_latents = torch.concat([conditioning_latents, masks], 1)

                # 4. Sample random noise and timesteps - this is the forward diffusion process
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # 5. Add noise to clean latents according to timestep (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 6. Get text embeddings for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # 7. Get BrushNet predictions (spatial conditioning features)
                down_block_res_samples, mid_block_res_sample, up_block_res_samples = brushnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    brushnet_cond=conditioning_latents,
                    return_dict=False,
                )

                # 8. Predict noise using UNet with BrushNet conditioning
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_add_samples=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_add_sample=mid_block_res_sample.to(dtype=weight_dtype),
                    up_block_add_samples=[
                        sample.to(dtype=weight_dtype) for sample in up_block_res_samples
                    ],
                    return_dict=False,
                )[0]

                # 9. Calculate target based on noise scheduler configuration
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise  # Predict the noise directly
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # 10. **CORE OBJECTIVE: MSE loss between predicted and actual noise**
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(brushnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging and checkpointing
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        brushnet = accelerator.unwrap_model(brushnet)
        brushnet.save_pretrained(args.output_dir)
        logger.info(f"Training completed! Model saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)