#!/usr/bin/env python3
"""
Parse GraphSAGE-T metrics from training log files
Quick way to get validation metrics when metrics.json is missing
"""

import re
import sys
from pathlib import Path

def parse_log_file(log_path):
    """Parse training log to extract metrics"""
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all epoch lines
    # Format: Epoch 246 | train_loss=0.1234 val_loss=0.5678 P=0.123 R=0.456 F1=0.789 ROC-AUC=0.987 AUPR=0.321 time=50.12s
    pattern = r'Epoch (\d+) \| train_loss=([\d.]+) val_loss=([\d.]+) P=([\d.]+) R=([\d.]+) F1=([\d.]+) ROC-AUC=([\d.]+) AUPR=([\d.]+) time=([\d.]+)s'
    
    matches = re.findall(pattern, content)
    
    if not matches:
        print(" No epoch data found in log file!")
        return None
    
    # Parse all epochs
    epochs = []
    for match in matches:
        epoch_data = {
            'epoch': int(match[0]),
            'train_loss': float(match[1]),
            'val_loss': float(match[2]),
            'precision': float(match[3]),
            'recall': float(match[4]),
            'f1': float(match[5]),
            'roc_auc': float(match[6]),
            'aupr': float(match[7]),
            'time': float(match[8]),
        }
        epochs.append(epoch_data)
    
    return epochs

def find_best_epoch(epochs):
    """Find epoch with best validation AUPR"""
    best = max(epochs, key=lambda x: x['aupr'])
    return best

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_graphsage_t_logs.py <path_to_log_file>")
        print("\nExample:")
        print("  python parse_graphsage_t_logs.py logs/HI-Small_Trans/graphsage_t/baseline_GraphSAGE_T_20251127_124109.txt")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    
    if not log_path.exists():
        print(f" Log file not found: {log_path}")
        sys.exit(1)
    
    print("="*70)
    print(f"Parsing GraphSAGE-T training log...")
    print(f"File: {log_path}")
    print("="*70)
    
    epochs = parse_log_file(log_path)
    
    if not epochs:
        print("❌ Failed to parse log file!")
        sys.exit(1)
    
    print(f"\n Found {len(epochs)} epochs of training data\n")
    
    # Find best epoch
    best = find_best_epoch(epochs)
    last = epochs[-1]
    
    print("="*70)
    print("BEST EPOCH (by Validation AUPR)")
    print("="*70)
    print(f"Epoch:          {best['epoch']}")
    print(f"Val AUPR:       {best['aupr']:.4f}")
    print(f"Val F1:         {best['f1']:.4f}")
    print(f"Val Precision:  {best['precision']:.4f}")
    print(f"Val Recall:     {best['recall']:.4f}")
    print(f"Val ROC-AUC:    {best['roc_auc']:.4f}")
    print(f"Val Loss:       {best['val_loss']:.4f}")
    print(f"Train Loss:     {best['train_loss']:.4f}")
    
    print("\n" + "="*70)
    print("LAST EPOCH (Early Stopping)")
    print("="*70)
    print(f"Epoch:          {last['epoch']}")
    print(f"Val AUPR:       {last['aupr']:.4f}")
    print(f"Val F1:         {last['f1']:.4f}")
    print(f"Val Precision:  {last['precision']:.4f}")
    print(f"Val Recall:     {last['recall']:.4f}")
    print(f"Val ROC-AUC:    {last['roc_auc']:.4f}")
    
    # Training curve summary
    print("\n" + "="*70)
    print("TRAINING CURVE SUMMARY")
    print("="*70)
    
    first = epochs[0]
    print(f"Starting AUPR (epoch 1):  {first['aupr']:.4f}")
    print(f"Best AUPR (epoch {best['epoch']}):     {best['aupr']:.4f}")
    print(f"Final AUPR (epoch {last['epoch']}):    {last['aupr']:.4f}")
    print(f"Improvement:              {(best['aupr'] - first['aupr']):.4f} (+{((best['aupr'] - first['aupr']) / first['aupr'] * 100):.1f}%)")
    
    # Show progression
    print("\n" + "="*70)
    print("AUPR PROGRESSION (Every 50 Epochs)")
    print("="*70)
    
    for i, epoch_data in enumerate(epochs):
        if epoch_data['epoch'] % 50 == 0 or epoch_data['epoch'] == best['epoch'] or epoch_data == last:
            marker = ""
            if epoch_data['epoch'] == best['epoch']:
                marker = " ← BEST"
            elif epoch_data == last:
                marker = " ← STOPPED"
            
            print(f"Epoch {epoch_data['epoch']:3d}: AUPR={epoch_data['aupr']:.4f}, F1={epoch_data['f1']:.4f}{marker}")
    
    print("\n" + "="*70)
    print("  NOTE: These are VALIDATION metrics from training")
    print("    For TEST metrics, use the recovery script!")
    print("="*70)
    
    # Estimate test performance
    print("\n" + "="*70)
    print("ESTIMATED TEST PERFORMANCE")
    print("="*70)
    print(f"Expected Test AUPR: ~{best['aupr'] * 0.95:.4f} - {best['aupr']:.4f}")
    print("(Test performance is typically 0-5% lower than validation)")
    print("\nFor exact test metrics, run:")
    print(f"  python recover_graphsage_t_metrics.py --results_dir results/HI-Small_Trans/graphsage_t")
    print("="*70)

if __name__ == "__main__":
    main()