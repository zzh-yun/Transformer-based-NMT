#!/usr/bin/env python3
"""
Visualize training metrics from log file
"""

import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional


def parse_log_file(log_path: str) -> Dict[str, List[float]]:
    """
    Parse training log file and extract metrics
    
    Expected log format:
    Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU-1: {bleu_1:.4f} | BLEU-2: {bleu_2:.4f} | BLEU-3: {bleu_3:.4f} | BLEU-4: {bleu_4:.4f}
    """
    epochs = []
    train_losses = []
    val_losses = []
    perplexities = []
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    learning_rates = []
    
    # Pattern to match the epoch log line (new standardized format)
    # Epoch {epoch}, Step {step}, Loss: {loss:.4f}, Perplexity: {perplexity:.4f}, Learning_Rate: {lr:.2e}, BLEU-1: {bleu_1:.4f}, BLEU-2: {bleu_2:.4f}, BLEU-3: {bleu_3:.4f}, BLEU-4: {bleu_4:.4f}
    epoch_pattern = re.compile(
        r'Epoch\s+(\d+),\s+Step\s+(\d+),\s+Loss:\s+([\d.]+),\s+Perplexity:\s+([\d.]+),\s+Learning_Rate:\s+([\d.e+-]+),\s+'
        r'BLEU-1:\s+([\d.]+),\s+BLEU-2:\s+([\d.]+),\s+BLEU-3:\s+([\d.]+),\s+BLEU-4:\s+([\d.]+)'
    )
    
    # Pattern to match step log line (simplified format)
    # Step {step}, Loss: {loss:.4f}
    step_pattern = re.compile(
        r'Step\s+(\d+),\s+Loss:\s+([\d.]+)'
    )
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # Try to match epoch log (new standardized format)
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            step = int(epoch_match.group(2))
            loss = float(epoch_match.group(3))
            perplexity = float(epoch_match.group(4))
            lr_str = epoch_match.group(5)
            bleu_1 = float(epoch_match.group(6))*100
            bleu_2 = float(epoch_match.group(7))*100
            bleu_3 = float(epoch_match.group(8))*100
            bleu_4 = float(epoch_match.group(9))*100
            
            epochs.append(epoch)
            train_losses.append(loss)
            val_losses.append(None)  # Val loss not in new format
            perplexities.append(perplexity)
            bleu_1_scores.append(bleu_1)
            bleu_2_scores.append(bleu_2)
            bleu_3_scores.append(bleu_3)
            bleu_4_scores.append(bleu_4)
            
            # Parse learning rate
            try:
                learning_rates.append(float(lr_str))
            except ValueError:
                learning_rates.append(None)
        
        # Try to match step log (for additional step-level loss tracking if needed)
        step_match = step_pattern.search(line)
        if step_match:
            # Step logs are for intermediate tracking, not epoch-level metrics
            pass
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'perplexities': perplexities,
        'bleu_1_scores': bleu_1_scores,
        'bleu_2_scores': bleu_2_scores,
        'bleu_3_scores': bleu_3_scores,
        'bleu_4_scores': bleu_4_scores,
        'learning_rates': learning_rates
    }


def visualize_metrics(data: Dict[str, List[float]], output_dir: str):
    """
    Create separate visualization plots from parsed data
    Generates 4 separate PNG files: loss, perplexity, bleu, learning_rate
    """
    epochs = data['epochs']
    
    if not epochs:
        print("Warning: No epoch data found in log file!")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_losses'], label='Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_path = output_path / 'training_loss.png'
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"Training Loss plot saved to: {loss_path}")
    plt.close()
    
    # 2. Perplexity
    if data['perplexities']:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, data['perplexities'], label='Perplexity', color='green', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Perplexity', fontsize=12)
        plt.title('Perplexity', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        perplexity_path = output_path / 'perplexity.png'
        plt.savefig(perplexity_path, dpi=300, bbox_inches='tight')
        print(f"Perplexity plot saved to: {perplexity_path}")
        plt.close()
    
    # 3. BLEU Scores
    plt.figure(figsize=(10, 6))
    if data['bleu_1_scores']:
        plt.plot(epochs, data['bleu_1_scores'], label='BLEU-1', color='#1f77b4', linewidth=2, alpha=0.8)
    if data['bleu_2_scores']:
        plt.plot(epochs, data['bleu_2_scores'], label='BLEU-2', color='#ff7f0e', linewidth=2, alpha=0.8)
    if data['bleu_3_scores']:
        plt.plot(epochs, data['bleu_3_scores'], label='BLEU-3', color='#2ca02c', linewidth=2, alpha=0.8)
    if data['bleu_4_scores']:
        plt.plot(epochs, data['bleu_4_scores'], label='BLEU-4', color='#d62728', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title('BLEU Scores', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    bleu_path = output_path / 'bleu_scores.png'
    plt.savefig(bleu_path, dpi=300, bbox_inches='tight')
    print(f"BLEU Scores plot saved to: {bleu_path}")
    plt.close()
    
    # 4. Learning Rate
    lr_values = [lr for lr in data['learning_rates'] if lr is not None]
    if lr_values:
        lr_epochs = [epochs[i] for i, lr in enumerate(data['learning_rates']) if lr is not None]
        plt.figure(figsize=(10, 6))
        plt.plot(lr_epochs, lr_values, label='Learning Rate', color='purple', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        lr_path = output_path / 'learning_rate.png'
        plt.savefig(lr_path, dpi=300, bbox_inches='tight')
        print(f"Learning Rate plot saved to: {lr_path}")
        plt.close()
    else:
        print("Warning: No Learning Rate data found, skipping Learning Rate plot")


def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics from log file')
    parser.add_argument('log_file', type=str, help='Path to training log file')
    parser.add_argument('-o', '--output', type=str, default='training_plots',
                       help='Output directory for plots (default: training_plots)')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    print(f"Parsing log file: {log_path}")
    data = parse_log_file(str(log_path))
    
    if not data['epochs']:
        print("Error: No epoch data found in log file. Please check the log format.")
        return
    
    print(f"Found {len(data['epochs'])} epochs")
    print(f"Epoch range: {min(data['epochs'])} - {max(data['epochs'])}")
    
    visualize_metrics(data, args.output)


if __name__ == '__main__':
    main()

