#!/usr/bin/env python3
"""
BYOL Analysis for macOS with Apple Silicon GPU Support
Modified version optimized for macOS with MPS (Metal Performance Shaders) support
"""

import os
import sys
import torch
from tqdm import tqdm

# Check and configure MPS (Apple Silicon GPU) support
def setup_device():
    """Setup device with MPS support for Apple Silicon"""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ Using Apple Silicon GPU (MPS): {device}")
        # Set memory fraction to avoid OOM on limited GPU memory
        torch.mps.set_per_process_memory_fraction(0.8)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using NVIDIA GPU: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU: {device}")
        # Optimize CPU performance
        torch.set_num_threads(os.cpu_count())

    return device

# Import the main analysis class after device setup
sys.path.insert(0, os.path.dirname(__file__))
from byol_cluster_analysis import BYOLClusterAnalysis, load_config, create_default_config, main as original_main

class BYOLMacOSAnalysis(BYOLClusterAnalysis):
    """macOS-optimized version of BYOL analysis"""

    def __init__(self, config):
        # Create output directory before calling parent __init__ (which sets up logging)
        from pathlib import Path
        output_path = Path(config['data']['output_path'])
        output_path.mkdir(parents=True, exist_ok=True)

        # Override device selection for macOS
        super().__init__(config)
        self.device = setup_device()
        self.logger.info(f"macOS Analysis initialized on device: {self.device}")

        # Adjust batch sizes for macOS/MPS limitations
        if self.device.type == 'mps':
            # MPS has memory limitations, reduce batch sizes
            original_batch_size = self.config['training']['batch_size']
            self.config['training']['batch_size'] = min(16, original_batch_size)

            original_inf_batch_size = self.config['inference']['batch_size']
            self.config['inference']['batch_size'] = min(32, original_inf_batch_size)

            if original_batch_size != self.config['training']['batch_size']:
                self.logger.info(f"Adjusted training batch size for MPS: {original_batch_size} -> {self.config['training']['batch_size']}")
            if original_inf_batch_size != self.config['inference']['batch_size']:
                self.logger.info(f"Adjusted inference batch size for MPS: {original_inf_batch_size} -> {self.config['inference']['batch_size']}")

    def setup_model(self):
        """Setup BYOL model with macOS optimizations"""
        self.logger.info("Setting up BYOL model for macOS...")

        # Import required modules
        from torchvision import models, transforms
        from torch import nn
        from byol_pytorch import BYOL

        # Data augmentations
        transform1 = nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        )

        transform2 = nn.Sequential(
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        )

        # Base model - use smaller model for MPS if needed
        if self.device.type == 'mps':
            # Use ResNet18 for better MPS compatibility
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # BYOL learner
        self.learner = BYOL(
            resnet,
            image_size=self.config['model']['image_size'],
            hidden_layer='avgpool',
            projection_size=self.config['model']['projection_size'],
            projection_hidden_size=self.config['model']['projection_hidden_size'],
            moving_average_decay=self.config['model']['moving_average_decay'],
            use_momentum=True,
            augment_fn=transform1,
            augment_fn2=transform2
        ).to(self.device)

        self.logger.info(f"BYOL model setup complete on {self.device}")
        return self.learner

    def train_model(self):
        """Train BYOL model with macOS optimizations"""
        self.logger.info("Starting BYOL training (macOS optimized)...")

        if self.learner is None:
            self.setup_model()

        from torch.optim import Adam        
        import torch

        optimizer = Adam(
            self.learner.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

        # Setup checkpointing
        checkpoint_path = self.output_path / 'model_checkpoint.pt'
        start_epoch = 0

        # Resume from checkpoint if exists
        if checkpoint_path.exists() and self.config['training']['resume']:
            self.logger.info("Loading checkpoint...")
            try:
                # Try with weights_only=False for our trusted checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.learner.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                self.logger.info(f"Resumed from epoch {start_epoch}")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
                self.logger.info("Starting training from scratch...")
                start_epoch = 0

        # Training loop with MPS error handling
        num_epochs = self.config['training']['num_epochs']
        save_interval = self.config['training']['save_interval']

        for epoch in tqdm(range(start_epoch, num_epochs), desc="Training BYOL (macOS)"):
            try:
                images = self.sample_unlabelled_images()
                loss = self.learner(images)

                optimizer.zero_grad()
                loss.backward()

                # Handle MPS gradient issues
                if self.device.type == 'mps':
                    # Clip gradients for MPS stability
                    torch.nn.utils.clip_grad_norm_(self.learner.parameters(), max_norm=1.0)

                optimizer.step()
                self.learner.update_moving_average()

                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

                # Save checkpoint
                if (epoch + 1) % save_interval == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.learner.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'config': self.config,
                        'device': str(self.device)
                    }
                    torch.save(checkpoint, checkpoint_path)
                    self.logger.info(f"Checkpoint saved at epoch {epoch}")

            except RuntimeError as e:
                if "MPS" in str(e):
                    self.logger.error(f"MPS error at epoch {epoch}: {e}")
                    self.logger.info("Trying to continue with CPU fallback...")
                    self.device = torch.device('cpu')
                    self.learner = self.learner.to(self.device)
                    continue
                else:
                    raise e

        # Save final model
        final_model_path = self.output_path / 'byol_final_model.pt'
        torch.save({
            'model_state_dict': self.learner.state_dict(),
            'config': self.config,
            'training_complete': True,
            'device': str(self.device)
        }, final_model_path)

        self.logger.info("Training completed successfully (macOS)")

    def extract_embeddings(self):
        """Extract embeddings with macOS optimizations"""
        self.logger.info("Extracting embeddings (macOS optimized)...")

        if self.learner is None:
            self.logger.info("Loading trained model...")
            model_path = self.output_path / 'byol_final_model.pt'
            if not model_path.exists():
                raise FileNotFoundError(f"Trained model not found: {model_path}")

            self.setup_model()
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.learner.load_state_dict(checkpoint['model_state_dict'])

        self.learner.eval()
        all_embeddings = []

        batch_size = self.config['inference']['batch_size']
        num_batches = (len(self.images) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Extracting embeddings (macOS)"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.images))

                batch = torch.tensor(
                    self.images[start_idx:end_idx],
                    dtype=torch.float32
                ).to(self.device)

                try:
                    _, embeddings = self.learner(batch, return_embedding=True)
                    all_embeddings.append(embeddings.cpu().numpy())
                except RuntimeError as e:
                    if "MPS" in str(e) and self.device.type == 'mps':
                        self.logger.warning(f"MPS error during embedding extraction: {e}")
                        # Fallback to CPU
                        batch = batch.cpu()
                        self.learner = self.learner.cpu()
                        self.device = torch.device('cpu')
                        _, embeddings = self.learner(batch, return_embedding=True)
                        all_embeddings.append(embeddings.cpu().numpy())
                    else:
                        raise e

        import numpy as np
        self.embeddings = np.vstack(all_embeddings)

        # Save embeddings
        embeddings_path = self.output_path / 'embeddings.npy'
        np.save(embeddings_path, self.embeddings)

        self.logger.info(f"Extracted embeddings shape: {self.embeddings.shape}")
        return self.embeddings


def main():
    """macOS-specific main function"""
    import argparse
    from pathlib import Path
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser(description='BYOL Analysis for macOS with MPS Support')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--mode', type=str, choices=['train', 'analyze', 'full'],
                       default='full', help='Analysis mode')
    parser.add_argument('--data-path', type=str, help='Path to input data')
    parser.add_argument('--output-path', type=str, help='Path to output directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        # Try to load byol_config.yaml from script directory first
        default_config_path = Path(__file__).parent / 'byol_config.yaml'
        if default_config_path.exists():
            config = load_config(str(default_config_path))
        else:
            config = create_default_config()

    # Override config with command line arguments
    if args.data_path:
        config['data']['input_path'] = args.data_path
    if args.output_path:
        config['data']['output_path'] = args.output_path
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.resume:
        config['training']['resume'] = args.resume

    # Convert paths to Path objects
    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    # Initialize macOS analysis
    analysis = BYOLMacOSAnalysis(config)

    # Run analysis based on mode
    try:
        if args.mode == 'full':
            analysis.run_full_pipeline()
        elif args.mode == 'train':
            analysis.load_data()
            analysis.train_model()
        elif args.mode == 'analyze':
            analysis.load_data()
            analysis.extract_embeddings()
            analysis.compute_pca_umap()
            analysis.compute_similarity_analysis()
            analysis.create_visualizations()
            analysis.compute_labeled_distance_metrics()

        print("✅ macOS Analysis completed successfully!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()