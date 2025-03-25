import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from comet_ml import Experiment
from PIL import Image
import os
from diffusers import StableDiffusionPipeline
from src.ddpm_scheduler import ConditionalDDPMScheduler, CustomDDPMScheduler
from diffusers import DDPMScheduler, UNet2DConditionModel
from src.pipelines.stable_diffusion_pipeline import CustomStableDiffusionPipeline
from src.pipelines.ddpm_pipeline import CustomDDPMPipeline
import numpy as np
import matplotlib.pyplot as plt
from src.trainers.ddpm_trainer import CorrelatedNoiseDDPMTrainer
from src.datasets import CityscapesDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for training")

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--lr_scheduler_step', type=int, default=4, help="Step of the lr scheduler.")
    parser.add_argument('--lr_scheduler_gamma', type=float, default=41.0, help="Gamma of the lr scheduler.") 
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")   
    parser.add_argument('--image_size', type=int, default=64, help="Size of (resized) input images")
    parser.add_argument('--output_dir', type=str, default="results_ce", help="Directory to store the results")
    parser.add_argument('--img_dir', type=str, default="rdata", help="Directory where images are stored")
    parser.add_argument('--data_base_dir', type=str, default="", help="Data and datalist directory")
    parser.add_argument("--keep_ratio", type=float, default=1.0, help="Ratio of the dataset to keep. Useful for training with smaller parts of the dataset.")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--default_root_dir', type=str, default="", help="Default root directory for Pytorch Lightning")
    parser.add_argument('--comet_ml_project_name', type=str, default="patch-synthetic", help="Project name, useful for Comet ML logging.")
    parser.add_argument('--comet_ml_exp_name', type=str, default="", help="Experiment name name, useful for Comet ML logging.")
    parser.add_argument('--comet_ml_workspace', type=str, default="gonzaq94", help="Comet ML workspace for logging.")


    return parser.parse_args()

args = parse_args()
os.environ["COMET_LOG_DIR"] = os.path.join(os.getenv("WORKDIR"), "comet_logs")

def main():

    # initialize Comet experiment
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=args.comet_ml_project_name,
        workspace=args.comet_ml_workspace
    )

    if len(args.comet_ml_exp_name) > 0:
        experiment.set_name(args.comet_ml_exp_name)

    comet_logger = CometLogger(
        api_key=os.getenv("COMET_API_KEY"), 
        project_name=args.comet_ml_project_name, 
        workspace=args.comet_ml_workspace,
        experiment_name=args.comet_ml_exp_name if len(args.comet_ml_exp_name) > 0 else None
    )

    progress_bar = TQDMProgressBar()

    # Data directories (Replace these with actual paths)
    data_pth = "data/Cityscapes"
    img_dir = os.path.join(data_pth, "leftImg8bit_trainvaltest", "leftImg8bit")
    mask_dir = os.path.join(data_pth, "gtFine_trainvaltest", "gtFine")

    # Define datasets and dataloader
    transform_train_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_train_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = CityscapesDataset(args.img_dir, mask_dir, "train", transform_image, transform_mask)
    val_dataset = CityscapesDataset(args.img_dir, mask_dir, "val", transform_image, transform_mask)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define diffusion model

    model = CorrelatedNoiseDDPMTrainer(
        model_name="google/ddpm-cat-256", 
        lr=1e-4, 
        noise_corr = 0.1
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best_val_map",
        save_top_k=1,
        monitor="val_map",
        mode="max",
    )

    trainer = pl.Trainer(
        default_root_dir=args.default_root_dir,
        max_epochs=args.num_epochs,
        val_check_interval=1.0,
        logger=comet_logger,
        callbacks=[progress_bar, checkpoint_callback],
        log_every_n_steps=1,
        )

    # Perform an evaluation before training starts
    valid_results = trainer.validate(model, val_dataloader)
    experiment.log_metrics({f"val_{k}": v for k, v in valid_results[0].items()}, step=0)

    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
