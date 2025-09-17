import sys
import os
import numpy as np

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the optimized model
from Luscious_Model.Lucious_toroidal_optimized import LuciousToroidalModel

def test_basic_functionality():
    """Test basic functionality of the optimized model."""
    print("Testing basic functionality...")
    
    # Create model with default configuration
    model = LuciousToroidalModel()
    
    # Load default data
    model.load_data()
    
    # Check if vocabulary was created
    print(f"Vocabulary size: {len(model.vocab)}")
    print(f"First 10 vocabulary items: {model.vocab[:10]}")
    
    # Check if sequences were created
    print(f"Number of sequences: {len(model.sequences)}")
    print(f"First sequence: {model.sequences[0][:10]}...")
    
    # Test token to phasor conversion
    sequence = model.sequences[0][:10]
    x_real, x_imag = model.token_to_phasor(sequence)
    print(f"Phasor shape: ({len(x_real)}, {len(x_imag)})")
    
    # Test phasor to token conversion
    tokens = model.phasor_to_token(x_real, x_imag)
    print(f"Converted tokens: {tokens[:5]}...")
    
    print("Basic functionality test completed.\n")

def test_training():
    """Test training functionality of the optimized model."""
    print("Testing training functionality...")
    
    # Create model with smaller configuration for faster testing
    config = {
        'sequence_length': 16,
        'batch_size': 4,
        'embedding_dim': 8,
        'max_iterations': 10
    }
    model = LuciousToroidalModel(config)
    
    # Load default data
    model.load_data()
    
    # Train for a single batch with fewer passes
    batch = model.sequences[:4]
    metrics = model.train_batch(batch, num_passes=2)
    
    # Check metrics
    print(f"Training metrics: {metrics}")
    
    print("Training test completed.\n")

def test_text_generation():
    """Test text generation functionality of the optimized model."""
    print("Testing text generation functionality...")
    
    # Create model with smaller configuration
    config = {
        'sequence_length': 16,
        'batch_size': 4,
        'embedding_dim': 8,
        'max_iterations': 10
    }
    model = LuciousToroidalModel(config)
    
    # Load default data
    model.load_data()
    
    # Generate text from a prompt
    prompt = "Once upon a time"
    generated_text = model.generate_text(prompt, max_length=10)
    
    print(f"Prompt: '{prompt}'")
    print(f"Generated text: '{generated_text}'")
    
    print("Text generation test completed.\n")

def test_save_load():
    """Test save and load functionality of the optimized model."""
    print("Testing save and load functionality...")
    
    # Create model
    model = LuciousToroidalModel()
    
    # Load default data
    model.load_data()
    
    # Save model
    save_path = "test_model.pkl"
    model.save_model(save_path)
    
    # Load model
    loaded_model = LuciousToroidalModel.load_model(save_path)
    
    # Check if loaded model has the same vocabulary
    print(f"Original vocab size: {len(model.vocab)}")
    print(f"Loaded vocab size: {len(loaded_model.vocab)}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Removed test file: {save_path}")
    
    print("Save/load test completed.\n")

if __name__ == "__main__":
    print("Testing LuciousToroidalModel...")
    
    # Run tests
    test_basic_functionality()
    test_training()
    test_text_generation()
    test_save_load()
    
    print("All tests completed.")