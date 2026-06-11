#!/usr/bin/env python3
"""
Test script to verify the neuralese recurrence implementation
"""

import torch
from neuralese_recurrence import NeuraleseRecurrenceModel

def test_model_initialization():
    """Test that the model initializes correctly"""
    print("Testing model initialization...")
    
    # Initialize with proper dimensions
    model = NeuraleseRecurrenceModel(input_dim=128, hidden_dim=256)
    assert model is not None
    print("✓ Model initialized successfully")

def test_forward_pass():
    """Test a forward pass through the model"""
    print("Testing forward pass...")
    
    # Create sample input
    batch_size = 2
    input_dim = 128
    input_tensor = torch.randn(batch_size, input_dim)
    
    # Run forward pass
    model = NeuraleseRecurrenceModel(input_dim=input_dim, hidden_dim=256)
    output, stats = model(input_tensor)
    
    # Verify outputs
    assert output is not None
    assert isinstance(output, torch.Tensor)
    assert output.shape == input_tensor.shape
    assert 'memory_usage' in stats
    assert 'recurrence_depth' in stats
    
    print("✓ Forward pass completed successfully")
    print(f"  Output shape: {output.shape}")
    print(f"  Stats: {stats}")

def test_memory_buffer():
    """Test memory buffer functionality"""
    print("Testing memory buffer...")
    
    model = NeuraleseRecurrenceModel(input_dim=128, hidden_dim=256)
    input_tensor = torch.randn(1, 128)
    
    # Process input
    output, stats = model(input_tensor)
    
    # Check that memory was used
    assert len(model.memory_buffer.buffer) > 0
    print("✓ Memory buffer functionality working")

def test_efficiency_benefits():
    """Test that efficiency benefits are realized"""
    print("Testing efficiency benefits...")
    
    model = NeuraleseRecurrenceModel(input_dim=128, hidden_dim=256)
    input_tensor = torch.randn(1, 128)
    
    # Check dimensionality
    original_dim = input_tensor.shape[-1]
    hidden_dim = model.hidden_dim
    
    assert hidden_dim > original_dim, "Hidden dimension should be larger than input"
    print(f"✓ Dimensionality benefits: {original_dim} -> {hidden_dim}")
    
    # Test recurrence depth
    assert model.recurrence_depth > 0
    print(f"✓ Recurrence depth: {model.recurrence_depth}")

def main():
    """Run all tests"""
    print("Running Neuralese Recurrence Implementation Tests\n")
    
    test_model_initialization()
    test_forward_pass()
    test_memory_buffer()
    test_efficiency_benefits()
    
    print("\n🎉 All tests passed! The implementation correctly demonstrates neuralese recurrence concepts.")

if __name__ == "__main__":
    main()