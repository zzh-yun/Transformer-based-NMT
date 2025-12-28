import os
import time
from click import Option
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.cuda.amp as amp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

import hydra
from omegaconf import DictConfig
from typing import Optional
from src.model import create_model
from src.dataset import DataPreprocessor, create_data_loaders
from evaluation import BLEUEvaluator

# import debugpy
# debugpy.listen(17170)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")


class NMTTrainer:
    """Neural Machine Translation Trainer with multi-GPU support"""
    
    def __init__(self, cfg):
        self.config = cfg
        
        # Initialize distributed training FIRST (before setting device)
        # This ensures LOCAL_RANK is properly set from environment variables
        if self.config.DISTRIBUTED:
            self.setup_distributed()
        
        # Set device after distributed setup
        self.device = torch.device(f'cuda:{self.config.LOCAL_RANK}' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        self.log(cfg)
        # Load data
        self.setup_data()
        
        # Create model
        self.setup_model()
        # ### 检查模型参数
        # for name, param in self.model.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"NaN in param {name}")
        # Setup optimizer and scheduler (will be set up in setup_model if loading checkpoint)
        # Initialize here for the case when not loading checkpoint
        self.setup_optimizer()
        
        # Setup evaluator
        self.evaluator = BLEUEvaluator()
        
        # Training state
        self.global_step = 0
        self.best_bleu = 0.0
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
        self.bleu_1_scores = []
        self.bleu_2_scores = []
        self.bleu_3_scores = []
        self.learning_rates = []
        
        # Early stopping
        self.patience = getattr(self.config, 'EARLY_STOPPING_PATIENCE', 5)  # Default patience of 5 epochs
        self.min_delta = getattr(self.config, 'EARLY_STOPPING_MIN_DELTA', 0.0)  # Minimum change to qualify as improvement
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision scaler
        self.scaler = amp.GradScaler()
        
        # Initialize start_epoch (will be updated in setup_model if loading checkpoint)
        self.start_epoch = 1
        
        # Generate experiment name and setup checkpoint directory
        self.experiment_name = self._generate_experiment_name()
        self.experiment_checkpoint_dir = os.path.join(self.config.CHECKPOINT_DIR, self.experiment_name)
        os.makedirs(self.experiment_checkpoint_dir, exist_ok=True)
        
    def _generate_experiment_name(self):
        """Generate experiment name based on configuration"""
        # If manual experiment name is provided, use it
        if hasattr(self.config, 'EXPERIMENT_NAME') and self.config.EXPERIMENT_NAME is not None:
            exp_name = str(self.config.EXPERIMENT_NAME)
            if hasattr(self.config, 'EXPERIMENT_TAG') and self.config.EXPERIMENT_TAG is not None:
                exp_name = f"{exp_name}_{self.config.EXPERIMENT_TAG}"
            return exp_name
        
        # Auto-generate experiment name
        pos_encoding = getattr(self.config, 'POS_ENCODING_TYPE', 'absolute')
        norm_type = getattr(self.config, 'NORM_TYPE', 'layernorm')
        batch_size = getattr(self.config, 'BATCH_SIZE', 64)
        learning_rate = getattr(self.config, 'LEARNING_RATE', 1e-4)
        
        # Shorten names for readability
        pos_short = 'abs' if pos_encoding == 'absolute' else 'rel'
        norm_short = 'ln' if norm_type == 'layernorm' else 'rms'
        
        # Format learning rate (e.g., 1e-4 -> 1e4, 5e-5 -> 5e5)
        lr_str = f"{learning_rate:.0e}".replace('e-0', 'e').replace('e-', 'e').replace('+', '')
        if lr_str.startswith('1e'):
            lr_str = lr_str.replace('1e', 'e')
        
        exp_name = f"exp_{pos_short}_pos_{norm_short}_bs{batch_size}_lr{lr_str}"
        
        # Add tag if provided
        if hasattr(self.config, 'EXPERIMENT_TAG') and self.config.EXPERIMENT_TAG is not None:
            exp_name = f"{exp_name}_{self.config.EXPERIMENT_TAG}"
        
        return exp_name
        
    def setup_distributed(self):
        """Setup distributed training"""
        # Ensure LOCAL_RANK is set from environment variable
        if 'LOCAL_RANK' in os.environ:
            self.config.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        
        # Validate LOCAL_RANK is within valid range
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，无法进行分布式训练")
        
        num_gpus = torch.cuda.device_count()
        if self.config.LOCAL_RANK >= num_gpus:
            raise RuntimeError(
                f"LOCAL_RANK={self.config.LOCAL_RANK} 超出可用GPU范围！\n"
                f"可用GPU数量: {num_gpus} (索引范围: 0-{num_gpus-1})\n"
                f"请检查:\n"
                f"  1. 系统实际可用的GPU数量\n"
                f"  2. CUDA_VISIBLE_DEVICES环境变量是否限制了可见GPU\n"
                f"  3. 是否使用了正确的torchrun命令（--nproc_per_node应与可用GPU数量匹配）"
            )
        
        # Set CUDA device BEFORE initializing process group
        torch.cuda.set_device(self.config.LOCAL_RANK)
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        
        # Verify device is set correctly
        if self.config.LOCAL_RANK == 0:
            print(f"分布式训练初始化完成: LOCAL_RANK={self.config.LOCAL_RANK}, GPU设备={self.config.LOCAL_RANK}, 总GPU数={num_gpus}")
        
    def setup_logging(self):
        """Setup logging"""
        if self.config.LOCAL_RANK == 0:
            # os.makedirs(self.config.LOGS_DIR, exist_ok=True)
            # logging.basicConfig(
            #     level=logging.INFO,
            #     format='%(asctime)s - %(levelname)s - %(message)s',
            #     handlers=[
            #         logging.FileHandler(os.path.join(self.config.LOGS_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            #         logging.StreamHandler()
            #     ]
            # )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
    
    def log(self, message):
        """Log message only from rank 0"""
        if self.logger:
            self.logger.info(message)
    
    def setup_data(self):
        """Setup data loaders"""
        self.log("Setting up data...")
        
        # Data preprocessing
        self.preprocessor = DataPreprocessor(self.config)
        
        # Try to load existing vocabularies from experiment checkpoint dir or base checkpoint dir
        vocab_dir = getattr(self, 'experiment_checkpoint_dir', self.config.CHECKPOINT_DIR)
        if not self.preprocessor.load_vocabularies(vocab_dir):
            # Fall back to base checkpoint directory
            if not self.preprocessor.load_vocabularies(self.config.CHECKPOINT_DIR):
            # Create new datasets and vocabularies
                train_dataset, val_dataset, test_dataset = self.preprocessor.create_datasets(
                    self.config.DATA_DIR
                )
                self.preprocessor.save_vocabularies(self.config.CHECKPOINT_DIR)
        else:
            # Load datasets with existing vocabularies
            train_dataset, val_dataset, test_dataset = self.preprocessor.create_datasets(
                self.config.DATA_DIR
            )
        
        # Create data loaders with custom collate function
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            distributed=self.config.DISTRIBUTED
        )
        
        self.log(f"Training batches: {len(self.train_loader)}")
        self.log(f"Validation batches: {len(self.val_loader)}")
        self.log(f"test batches: {len(self.test_loader)}")
    
    def _load_model(self, checkpoint_path):
        """Load model from checkpoint (like inference.py)"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.log(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model_state = checkpoint['model_state_dict']
        if self.config.DISTRIBUTED:
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
        
        self.log(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}, BLEU: {checkpoint.get('bleu_score', 'N/A')}")
        
        return checkpoint
    
    def setup_model(self):
        """Setup model"""
        self.log("Setting up model...")
        
        src_vocab_size = len(self.preprocessor.src_vocab)
        tgt_vocab_size = len(self.preprocessor.tgt_vocab)
        
        # Get position encoding and normalization types from config
        pos_encoding_type = getattr(self.config, 'POS_ENCODING_TYPE', 'absolute')
        norm_type = getattr(self.config, 'NORM_TYPE', 'layernorm')
        
        self.log(f"Position encoding type: {pos_encoding_type}")
        self.log(f"Normalization type: {norm_type}")
        
        # Create model
        self.model = create_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            device=self.device,
            pos_encoding_type=pos_encoding_type,
            norm_type=norm_type,
            d_model=self.config.D_MODEL,
            nhead=self.config.NHEAD,
            num_encoder_layers=self.config.NUM_ENCODER_LAYERS,
            num_decoder_layers=self.config.NUM_DECODER_LAYERS,
            dim_feedforward=self.config.DIM_FEEDFORWARD,
            dropout=self.config.DROPOUT,
            max_len=self.config.MAX_LEN
        )
        
        # Setup distributed model
        if self.config.DISTRIBUTED:
            self.model = DDP(self.model, device_ids=[self.config.LOCAL_RANK])
        
        # Loss function with label smoothing
        label_smoothing = getattr(self.config, 'LABEL_SMOOTHING', 0.1)  # Default 0.1
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.preprocessor.tgt_vocab.PAD_IDX,
            label_smoothing=label_smoothing
        )
        if self.config.LOCAL_RANK == 0:
            self.log(f"Using label smoothing: {label_smoothing}")
        
        self.log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Load model from checkpoint if specified (like inference.py)
        resume_from_checkpoint = getattr(self.config, 'resume_from_checkpoint', None)
        if resume_from_checkpoint is not None:
            self.checkpoint = self._load_model(resume_from_checkpoint)
        else:
            self.checkpoint = None
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=self.config.LOCAL_RANK != 0)   # tqdm 本身就是一个可迭代对象
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            src = batch['src'].to(self.device)  # (length, batch_size)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            # Create masks
            model_ref = self.model.module if self.config.DISTRIBUTED else self.model
            tgt_mask = model_ref.generate_square_subsequent_mask(tgt_input.size(0)).to(self.device)  # input: tgt_input.size, output:
            src_key_padding_mask = model_ref.create_padding_mask(src, self.preprocessor.src_vocab.PAD_IDX).float()
            tgt_key_padding_mask = model_ref.create_padding_mask(tgt_input, self.preprocessor.tgt_vocab.PAD_IDX).float()
            
            # print(f"tgt_mask shape: {tgt_mask.shape}, has NaN: {torch.isnan(tgt_mask).any()}")
            # print(f"src_key_padding_mask shape: {src_key_padding_mask.shape}, has NaN: {torch.isnan(src_key_padding_mask).any()}")
            # 检查是否存在nan值
            assert not torch.isnan(src).any(), "src contains NaN"
            assert not torch.isnan(tgt_input).any(), "tgt_input contains NaN" 

            # Forward pass
            self.optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True) 
            with amp.autocast():
                output = self.model(
                    src=src,
                    tgt=tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                assert not torch.isnan(output).any(), "the output has nan value"
                # Calculate loss
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # Check for NaN in loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                if self.config.LOCAL_RANK == 0:
                    print(f"Warning: NaN/Inf loss detected at step {self.global_step}, skipping this batch")
                self.optimizer.zero_grad()
                continue
            
            # Backward pass with error handling
            try:
                self.scaler.scale(loss).backward()
            except RuntimeError as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    if self.config.LOCAL_RANK == 0:
                        print(f"Warning: NaN/Inf in backward pass at step {self.global_step}, skipping this batch")
                    self.optimizer.zero_grad()
                    continue
                else:
                    raise
            
            # Check for NaN in gradients before unscaling
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        if self.config.LOCAL_RANK == 0:
                            print(f"Warning: NaN/Inf in grad of {name} at step {self.global_step}")
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                self.optimizer.zero_grad()
                continue
            
            self.scaler.unscale_(self.optimizer)
            
            # Check again after unscaling
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        if self.config.LOCAL_RANK == 0:
                            print(f"Warning: NaN/Inf in grad of {name} after unscaling, skipping this batch")
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                self.optimizer.zero_grad()
                continue
            
            # Calculate gradient norm before clipping with error handling
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.CLIP_GRAD)
                if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
                    if self.config.LOCAL_RANK == 0:
                        print(f"Warning: NaN/Inf gradient norm, skipping this batch")
                    self.optimizer.zero_grad()
                    continue
            except RuntimeError as e:
                if self.config.LOCAL_RANK == 0:
                    print(f"Warning: Error in gradient clipping, skipping this batch: {e}")
                self.optimizer.zero_grad()
                continue
            
            # Calculate parameter update norm
            param_update_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_update_norm += (param.grad * self.optimizer.param_groups[0]['lr']).norm().item() ** 2
            param_update_norm = param_update_norm ** 0.5
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            if self.config.LOCAL_RANK == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'grad_norm': f'{grad_norm:.2f}'
                })
            
            # Log intermediate results (simplified format)
            if self.global_step % self.config.LOG_INTERVAL == 0 and self.config.LOCAL_RANK == 0:
                self.log(f'Step {self.global_step}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        all_sources = []  # 用于存储源句子
        sample_count = 0  # 用于计数，只输出前两个测试用例
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', disable=self.config.LOCAL_RANK != 0)
            
            for batch in pbar:
                # Move to device
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device) 
                tgt_output = batch['tgt_output'].to(self.device)
                
                # Create masks for loss calculation (still use teacher forcing for loss)
                model_ref = self.model.module if self.config.DISTRIBUTED else self.model
                tgt_mask = model_ref.generate_square_subsequent_mask(tgt_input.size(0)).to(self.device)
                src_key_padding_mask = model_ref.create_padding_mask(src, self.preprocessor.src_vocab.PAD_IDX).float()
                tgt_key_padding_mask = model_ref.create_padding_mask(tgt_input, self.preprocessor.tgt_vocab.PAD_IDX).float()
                
                # Forward pass for loss calculation (using teacher forcing)
                output = self.model(
                    src=src,
                    tgt=tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                # Calculate loss
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                total_loss += loss.item()
                
                # Get predictions using autoregressive decoding (not teacher forcing)
                batch_size = src.size(1)
                # Use smaller max_len for validation to speed up
                val_max_len = min(self.config.MAX_LEN, getattr(self.config, 'VAL_MAX_LEN', 64))
                
                # Initialize decoder input with SOS token
                decoder_input = torch.full((1, batch_size), self.preprocessor.tgt_vocab.SOS_IDX, 
                                         dtype=torch.long, device=self.device)
                
                # Track which samples have finished (reached EOS)
                finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                
                # Get sampling parameters
                temperature = getattr(self.config, 'TEMPERATURE', 1.0)
                repetition_penalty = getattr(self.config, 'REPETITION_PENALTY', 1.0)
                
                # Autoregressive decoding
                for step in range(val_max_len):
                    # Early exit if all samples are finished
                    if torch.all(finished):
                        break
                    
                    # Create masks for current decoder input
                    tgt_mask_dec = model_ref.generate_square_subsequent_mask(decoder_input.size(0)).to(self.device)
                    tgt_key_padding_mask_dec = model_ref.create_padding_mask(decoder_input, self.preprocessor.tgt_vocab.PAD_IDX).float()
                    
                    # Forward pass
                    output_dec = self.model(
                        src=src,
                        tgt=decoder_input,
                        tgt_mask=tgt_mask_dec,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask_dec,
                        memory_key_padding_mask=src_key_padding_mask
                    )
                    
                    # Get next token with temperature sampling and repetition penalty
                    logits = output_dec[-1]  # (batch_size, vocab_size)
                    
                    # Apply temperature
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    # Optimized repetition penalty (vectorized, only process unfinished samples)
                    if repetition_penalty > 1.0:
                        for batch_idx in range(batch_size):
                            if not finished[batch_idx]:  # Only process unfinished samples
                                generated_tokens = decoder_input[1:, batch_idx]  # Skip SOS token
                                # Get unique tokens, excluding special tokens
                                unique_tokens = generated_tokens.unique()
                                unique_tokens = unique_tokens[
                                    (unique_tokens != self.preprocessor.tgt_vocab.PAD_IDX) &
                                    (unique_tokens != self.preprocessor.tgt_vocab.SOS_IDX) &
                                    (unique_tokens != self.preprocessor.tgt_vocab.EOS_IDX)
                                ]
                                if len(unique_tokens) > 0:
                                    # Vectorized operation: apply penalty to all unique tokens at once
                                    logits[batch_idx, unique_tokens] /= repetition_penalty
                    
                    # Sample from the distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).squeeze(-1)  # (batch_size,)
                    
                    # Mark samples that have reached EOS
                    eos_mask = (next_token == self.preprocessor.tgt_vocab.EOS_IDX)
                    finished = finished | eos_mask
                    
                    # Replace EOS with PAD for finished samples to avoid further generation
                    # This prevents the model from continuing to generate for finished sequences
                    next_token = torch.where(finished,
                                            torch.full_like(next_token, self.preprocessor.tgt_vocab.PAD_IDX),
                                            next_token)
                    
                    decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=0)
                
                # Convert decoder output to predictions (skip SOS token)
                pred = decoder_input[1:].transpose(0, 1)  # (batch_size, seq_len)
                
                # Convert to text
                pred_text = self.preprocessor.tgt_vocab.decode_batch(pred.cpu().numpy())
                ref_text = self.preprocessor.tgt_vocab.decode_batch(batch['tgt_output'].transpose(0,1).cpu().numpy())
                src_text = self.preprocessor.src_vocab.decode_batch(batch['src'].transpose(0,1).cpu().numpy())
                
                # Skip verbose test case output for cleaner logs
                
                all_predictions.extend(pred_text)
                all_references.extend([[r] for r in ref_text])
                all_sources.extend(src_text)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算不同n-gram的BLEU得分以便调试
        bleu_1 = self.evaluator.corpus_bleu(all_references, all_predictions, n_gram=1)
        bleu_2 = self.evaluator.corpus_bleu(all_references, all_predictions, n_gram=2)
        bleu_3 = self.evaluator.corpus_bleu(all_references, all_predictions, n_gram=3)
        bleu_4 = self.evaluator.corpus_bleu(all_references, all_predictions, n_gram=4)
        
        # Check for mode collapse (only log warnings)
        if self.config.LOCAL_RANK == 0:
            from collections import Counter
            all_pred_tokens = []
            for pred in all_predictions:
                all_pred_tokens.extend(pred.split())
            token_counter = Counter(all_pred_tokens)
            top_tokens = token_counter.most_common(10)
            total_tokens = len(all_pred_tokens)
            
            # 检查是否陷入模式崩溃（如果前3个token占超过50%）
            if total_tokens > 0:
                top3_ratio = sum(count for _, count in top_tokens[:3]) / total_tokens
                if top3_ratio > 0.5:
                    self.log(f"Warning: Mode collapse detected (top 3 tokens: {top3_ratio:.2%})")
        
        return avg_loss, bleu_1, bleu_2, bleu_3, bleu_4
    
    def save_checkpoint(self, epoch, bleu_score, is_best=False):
        """Save model checkpoint in experiment-specific directory"""
        if self.config.LOCAL_RANK == 0:
            # Use experiment-specific checkpoint directory
            os.makedirs(self.experiment_checkpoint_dir, exist_ok=True)
            
            model_state = self.model.module.state_dict() if self.config.DISTRIBUTED else self.model.state_dict()
            
            # Get experiment configuration
            pos_encoding_type = getattr(self.config, 'POS_ENCODING_TYPE', 'absolute')
            norm_type = getattr(self.config, 'NORM_TYPE', 'layernorm')
            
            checkpoint = {
                'epoch': epoch,
                'bleu_score': bleu_score,
                'best_bleu': self.best_bleu,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),  # Mixed precision scaler
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'bleu_scores': self.bleu_scores,
                'bleu_1_scores': self.bleu_1_scores,
                'bleu_2_scores': self.bleu_2_scores,
                'bleu_3_scores': self.bleu_3_scores,
                'learning_rates': self.learning_rates,
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
                'global_step': self.global_step,
                'src_vocab': self.preprocessor.src_vocab,  # Save vocabularies in checkpoint
                'tgt_vocab': self.preprocessor.tgt_vocab,
                # Save experiment configuration
                'experiment_name': self.experiment_name,
                'pos_encoding_type': pos_encoding_type,
                'norm_type': norm_type,
                'config': dict(self.config),  # Save full config
            }
            
            filename = os.path.join(self.experiment_checkpoint_dir, 'checkpoint_last.pth')
            torch.save(checkpoint, filename)
            
            if is_best:
                best_filename = os.path.join(self.experiment_checkpoint_dir, 'checkpoint_best.pth')
                torch.save(checkpoint, best_filename)
                self.log(f"Best checkpoint saved to {best_filename}")
    
    def plot_metrics(self):
        """Plot and save training metrics"""
        if self.config.LOCAL_RANK == 0:
            plt.figure(figsize=(18, 12))
            
            # Training Loss
            plt.subplot(2, 2, 1)
            plt.plot(self.train_losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
            
            # Validation Loss
            plt.subplot(2, 2, 2)
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Validation Loss')
            plt.grid(True)
            
            # BLEU Scores
            plt.subplot(2, 2, 3)
            if self.bleu_1_scores:
                plt.plot(self.bleu_1_scores, label='BLEU-1', alpha=0.7)
            if self.bleu_2_scores:
                plt.plot(self.bleu_2_scores, label='BLEU-2', alpha=0.7)
            if self.bleu_3_scores:
                plt.plot(self.bleu_3_scores, label='BLEU-3', alpha=0.7)
            if self.bleu_scores:
                plt.plot(self.bleu_scores, label='BLEU-4', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('BLEU Score')
            plt.title('BLEU Scores')
            plt.legend()
            plt.grid(True)

            # Learning Rate
            plt.subplot(2, 2, 4)
            plt.plot(self.learning_rates, label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('LR')
            plt.title('Learning Rate')
            plt.grid(True)

            plt.tight_layout()
            save_path = os.path.join(self.experiment_checkpoint_dir, 'training_metrics.png')
            plt.savefig(save_path)
            self.log(f"Metrics plot saved to {save_path}")
            plt.close()

    def train(self):
        """Main training loop"""
        # Load training state from checkpoint if available
        if hasattr(self, 'checkpoint') and self.checkpoint is not None:
            start_epoch = self.checkpoint['epoch'] + 1  # Start from next epoch
            self.best_bleu = self.checkpoint.get('best_bleu', 0.0)
            self.global_step = self.checkpoint.get('global_step', 0)
            self.best_val_loss = self.checkpoint.get('best_val_loss', float('inf'))
            self.patience_counter = self.checkpoint.get('patience_counter', 0)
            self.train_losses = self.checkpoint.get('train_losses', [])
            self.val_losses = self.checkpoint.get('val_losses', [])
            self.bleu_scores = self.checkpoint.get('bleu_scores', [])
            self.bleu_1_scores = self.checkpoint.get('bleu_1_scores', [])
            self.bleu_2_scores = self.checkpoint.get('bleu_2_scores', [])
            self.bleu_3_scores = self.checkpoint.get('bleu_3_scores', [])
            self.learning_rates = self.checkpoint.get('learning_rates', [])
            self.log(f"Resuming training from epoch {start_epoch}...")
            self.log(f"Best BLEU so far: {self.best_bleu:.4f}")
        else:
            start_epoch = 1
            self.log("Starting training from scratch...")
        
        for epoch in range(start_epoch, self.config.NUM_EPOCHS + 1):
            if self.config.DISTRIBUTED:
                self.train_loader.sampler.set_epoch(epoch)
                
            train_loss = self.train_epoch(epoch)
            val_loss, bleu_1, bleu_2, bleu_3, bleu_4 = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.bleu_scores.append(bleu_4)
            self.bleu_1_scores.append(bleu_1)
            self.bleu_2_scores.append(bleu_2)
            self.bleu_3_scores.append(bleu_3)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            if self.config.LOCAL_RANK == 0:
                # Calculate perplexity from loss
                perplexity = np.exp(train_loss)
                learning_rate = self.optimizer.param_groups[0]['lr']
                # Standardized log format: Epoch, Step, Loss, Perplexity, Learning_Rate, BLEU-1, BLEU-2, BLEU-3, BLEU-4
                self.log(f'Epoch {epoch}, Step {self.global_step}, Loss: {train_loss:.4f}, Perplexity: {perplexity:.4f}, Learning_Rate: {learning_rate:.2e}, BLEU-1: {bleu_1:.4f}, BLEU-2: {bleu_2:.4f}, BLEU-3: {bleu_3:.4f}, BLEU-4: {bleu_4:.4f}')
                
                is_best = bleu_4 > self.best_bleu
                if is_best:
                    self.best_bleu = bleu_4
                
                self.save_checkpoint(epoch, bleu_4, is_best)
                self.plot_metrics()

            # Update scheduler
            self.scheduler.step(bleu_4)
            
            # Early stopping check (based on validation loss)
            if self.config.LOCAL_RANK == 0:
                if val_loss < self.best_val_loss - self.min_delta:
                    # Validation loss improved
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    # No improvement
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.patience:
                    self.log(f"\n早停触发: 验证损失在 {self.patience} 个epoch内没有改善")
                    self.log(f"最佳验证损失: {self.best_val_loss:.4f}")
                    self.log(f"当前验证损失: {val_loss:.4f}")
                    break
            
        self.log("Training finished.")

@hydra.main(version_base='1.3', config_path='./configs', config_name='train.yaml')
def main(cfgs: DictConfig) -> Optional[float]:
    
    # Update config with distributed training settings
    # 优先从环境变量读取LOCAL_RANK（torchrun会自动设置）
    # 如果没有环境变量，则使用配置文件中的值
    cfgs.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", getattr(cfgs, 'local_rank', 0)))
    cfgs.DISTRIBUTED = cfgs.distributed or int(os.environ.get("WORLD_SIZE", 1)) > 1

    if cfgs.DISTRIBUTED:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        cfgs.BATCH_SIZE = cfgs.BATCH_SIZE // world_size
        print(f"分布式训练: 使用 {world_size} 个GPU, LOCAL_RANK={cfgs.LOCAL_RANK}, 每个GPU的批次大小={cfgs.BATCH_SIZE}")

    trainer = NMTTrainer(cfgs)
    trainer.train()

if __name__ == '__main__':
    main() 