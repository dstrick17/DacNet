import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars


def find_optimal_batch_size(model, trainset, device, num_workers, start_size=32, max_size=512):
    """Find the optimal batch size by testing increasingly larger sizes until OOM error occurs.
    
    Args:
        model: The model to find the optimal batch size for.
        trainset: The training set to find the optimal batch size for.
        device: The device to find the optimal batch size for.
        start_size: The starting batch size to test.
        max_size: The maximum batch size to test.
    
    Returns:
        The optimal batch size.
    
    Example:
        >>> optimal_batch_size = find_optimal_batch_size(model, trainset, device)
    """
    import time
    
    model = model.to(device)
    optimal_size = start_size
    times = {}
    
    print(f"Testing batch sizes on device: {device}")
    
    # Test different batch sizes
    batch_size = start_size
    while batch_size <= max_size:
        try:
            # Create a dataloader with the current batch size
            loader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers, pin_memory=True
            )
            
            # Time a few batches
            start_time = time.time()
            batch_count = 0
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                batch_count += 1
                if batch_count >= 5:  # Test with 5 batches
                    break
                    
            end_time = time.time()
            
            # Calculate throughput (images/second)
            elapsed = end_time - start_time
            throughput = (batch_size * batch_count) / elapsed
            times[batch_size] = throughput
            
            print(f"Batch size {batch_size}: {throughput:.2f} images/sec")
            
            # Update optimal size if this one worked
            optimal_size = batch_size
            
            # Increase batch size for next iteration
            batch_size *= 2
            
        except (RuntimeError, torch.cuda.OutOfMemoryError, torch.mps.OutOfMemoryError):
            # Memory error occurred, break the loop
            print(f"Memory error at batch size {batch_size}")
            break
    
    # Find the batch size with the best throughput
    best_batch_size = max(times.items(), key=lambda x: x[1])[0]
    print(f"\nOptimal batch size: {best_batch_size} (throughput: {times[best_batch_size]:.2f} images/sec)")
    
    return best_batch_size



