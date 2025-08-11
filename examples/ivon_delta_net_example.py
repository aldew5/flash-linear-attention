#!/usr/bin/env python3
"""
Example script demonstrating IVON DeltaNet vs Standard DeltaNet

This script shows how to use the IVON (Inverse Variance Online Newton) variant
of DeltaNet that replaces SGD with IVON for memory updates.
"""

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

# Import the IVON DeltaNet implementation
from fla.layers.ivon_delta_net import IVONDeltaNet
from fla.layers.delta_net import DeltaNet


def compare_delta_rule_implementations():
    """
    Compare standard DeltaNet with IVON DeltaNet
    """
    print("=== DeltaNet vs IVON DeltaNet Comparison ===\n")
    
    # Configuration
    batch_size = 2
    seq_len = 128
    hidden_size = 512
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    # Create input tensors
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    
    print(f"Input shape: {x.shape}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {head_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}\n")
    
    # Initialize models
    standard_deltanet = DeltaNet(
        mode='fused_recurrent',
        hidden_size=hidden_size,
        num_heads=num_heads,
        use_beta=True,
        use_gate=False,
        use_short_conv=True,
        conv_size=4,
        layer_idx=0
    ).cuda()
    
    ivon_deltanet = IVONDeltaNet(
        mode='ivon_fused_recurrent',
        hidden_size=hidden_size,
        num_heads=num_heads,
        use_beta=True,
        use_gate=False,
        use_short_conv=True,
        conv_size=4,
        ivon_momentum=0.9,
        ivon_epsilon=1e-8,
        layer_idx=0
    ).cuda()
    
    print("Model configurations:")
    print(f"Standard DeltaNet parameters: {sum(p.numel() for p in standard_deltanet.parameters()):,}")
    print(f"IVON DeltaNet parameters: {sum(p.numel() for p in ivon_deltanet.parameters()):,}")
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    
    # Standard DeltaNet
    with torch.no_grad():
        standard_output, _, _ = standard_deltanet(x)
        print(f"Standard DeltaNet output shape: {standard_output.shape}")
    
    # IVON DeltaNet
    with torch.no_grad():
        ivon_output, _, _ = ivon_deltanet(x)
        print(f"IVON DeltaNet output shape: {ivon_output.shape}")
    
    print()
    
    # Test with gradients (training mode)
    print("Testing with gradients (training mode)...")
    
    # Standard DeltaNet
    standard_output, _, _ = standard_deltanet(x)
    standard_loss = standard_output.sum()
    standard_loss.backward()
    
    # IVON DeltaNet
    ivon_output, _, _ = ivon_deltanet(x)
    ivon_loss = ivon_output.sum()
    ivon_loss.backward()
    
    print("✓ Both models successfully computed gradients")
    print()
    
    # Compare outputs
    output_diff = torch.abs(standard_output - ivon_output).mean()
    print(f"Average absolute difference between outputs: {output_diff:.6f}")
    
    if output_diff < 1e-3:
        print("✓ Outputs are very similar (as expected for similar architectures)")
    else:
        print("⚠ Outputs differ significantly (this might indicate implementation differences)")
    
    print()


def demonstrate_ivon_advantages():
    """
    Demonstrate the theoretical advantages of IVON over SGD
    """
    print("=== IVON vs SGD Theoretical Advantages ===\n")
    
    print("1. Adaptive Learning Rates:")
    print("   - SGD: Uses fixed learning rate β")
    print("   - IVON: Uses adaptive learning rate 1/variance_t")
    print("   - Advantage: IVON automatically adjusts learning rate based on gradient variance")
    print()
    
    print("2. Better Convergence Properties:")
    print("   - SGD: First-order optimization method")
    print("   - IVON: Second-order optimization method (approximates Newton's method)")
    print("   - Advantage: IVON typically converges faster and to better minima")
    print()
    
    print("3. Variance Tracking:")
    print("   - SGD: No memory of past gradients")
    print("   - IVON: Maintains exponential moving average of gradient variance")
    print("   - Advantage: IVON can adapt to changing gradient statistics")
    print()
    
    print("4. Memory Update Formula:")
    print("   - SGD Delta Rule: S_t = S_{t-1} + β * v_t * k_t^T")
    print("   - IVON Delta Rule: S_t = S_{t-1} + (1/variance_t) * v_t * k_t^T")
    print("   - Advantage: IVON uses inverse variance as preconditioner")
    print()


def performance_benchmark():
    """
    Simple performance benchmark
    """
    print("=== Performance Benchmark ===\n")
    
    # Configuration
    batch_size = 4
    seq_len = 256
    hidden_size = 1024
    num_heads = 16
    
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    
    # Initialize models
    standard_deltanet = DeltaNet(
        mode='fused_recurrent',
        hidden_size=hidden_size,
        num_heads=num_heads,
        use_beta=True,
        layer_idx=0
    ).cuda()
    
    ivon_deltanet = IVONDeltaNet(
        mode='ivon_fused_recurrent',
        hidden_size=hidden_size,
        num_heads=num_heads,
        use_beta=True,
        layer_idx=0
    ).cuda()
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = standard_deltanet(x)
            _ = ivon_deltanet(x)
    
    # Benchmark
    torch.cuda.synchronize()
    
    # Standard DeltaNet
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(10):
        with torch.no_grad():
            _ = standard_deltanet(x)
    end_time.record()
    torch.cuda.synchronize()
    standard_time = start_time.elapsed_time(end_time) / 10
    
    # IVON DeltaNet
    start_time.record()
    for _ in range(10):
        with torch.no_grad():
            _ = ivon_deltanet(x)
    end_time.record()
    torch.cuda.synchronize()
    ivon_time = start_time.elapsed_time(end_time) / 10
    
    print(f"Standard DeltaNet average time: {standard_time:.2f} ms")
    print(f"IVON DeltaNet average time: {ivon_time:.2f} ms")
    print(f"Speedup: {standard_time / ivon_time:.2f}x")
    print()


if __name__ == "__main__":
    print("IVON DeltaNet Example")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this on a GPU.")
        exit(1)
    
    try:
        compare_delta_rule_implementations()
        demonstrate_ivon_advantages()
        performance_benchmark()
        
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("This might be due to missing dependencies or CUDA issues.") 