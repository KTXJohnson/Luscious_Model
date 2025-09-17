# Lucious Toroidal Model for Chatbot Training

This directory contains the Lucious Toroidal Model, a mathematical model for text processing using toroidal transformations and phasor encoding. The model has been optimized for generalized chatbot training.

## Files

- `Lucious_toroidal.py` - Original implementation
- `Lucious_toroidal_optimized.py` - Optimized implementation for chatbot training
- `test_optimized_model.py` - Test script for the optimized implementation

## Optimizations for Chatbot Training

The optimized implementation includes the following improvements:

1. **Class-based Structure**: Organized code into a class-based structure for better maintainability and extensibility.
2. **Configurable Parameters**: All model parameters are configurable through a configuration dictionary.
3. **Improved Tokenization**: Added support for special tokens (`<PAD>`, `<UNK>`, `<START>`, `<END>`) and better text preprocessing.
4. **Batch Processing**: Added support for batch processing to improve training efficiency.
5. **Variable Sequence Length**: Support for longer sequences and padding/truncation to handle variable-length inputs.
6. **Text Generation**: Added a text generation function specifically designed for chatbot responses.
7. **Model Persistence**: Added functions to save and load models for continued training or deployment.
8. **External Data Loading**: Support for loading data from external files (JSON, text) for training on custom datasets.
9. **Improved Documentation**: Comprehensive docstrings and type hints for better code understanding.

## Repository Structure

The project is organized into the following directory structure:

```
Luscious_Model/
├── models/             # Model definitions
├── training/           # Training code
├── inference/          # Inference code
├── utils/              # Utility functions
└── tests/              # Test code
```

## GitHub Repository Setup

To create and push to a new GitHub repository:

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click the "+" icon in the top right corner and select "New repository"
3. Enter "Luscious_Model" as the repository name
4. Add a description (optional)
5. Choose whether the repository should be public or private
6. Do NOT initialize the repository with a README, .gitignore, or license
7. Click "Create repository"
8. After creating the repository, run the following command to push your local repository:

```
git push -u origin master
```

## Usage

### Basic Usage

```python
from Luscious_Model.Lucious_toroidal_optimized import LuciousToroidalModel

# Create model with default configuration
model = LuciousToroidalModel()

# Load default data
model.load_data()

# Train model
model.train(num_epochs=1, batch_size=8, num_passes=10)

# Generate text
prompt = "In a toroidal drift, nodules pulsed through zeta folds"
generated_text = model.generate_text(prompt, max_length=20)
print(f"Generated text: {generated_text}")

# Save model
model.save_model("lucious_toroidal_model.pkl")
```

### Custom Configuration

```python
# Create model with custom configuration
config = {
    'sequence_length': 128,
    'batch_size': 32,
    'vocab_size': 20000,
    'embedding_dim': 32,
    'learning_rate': 0.005,
    'max_iterations': 200,
    'f0': 440,
    'beta': 0.02,
    'gamma': 0.1,
    'sigma': 15,
    'phi': 1.6180339887,
    'spike_freqs': [0.618, 1.618, 2.618, 3.618, 4.618],
    'fib': [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
    'luc': [2, 1, 3, 4, 7, 11, 18, 29, 47, 76],
    'primes': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
    'zeros_t': [14.135, 21.022, 25.011, 30.425, 32.935]
}
model = LuciousToroidalModel(config)
```

### Loading Custom Data

```python
# Load data from a JSON file
model.load_data(data_path="path/to/data.json", max_vocab_size=20000)

# Load data from a text file
model.load_data(data_path="path/to/data.txt", max_vocab_size=20000)

# Load data from a list of dictionaries
data = [
    {"text": "This is a sample text for training."},
    {"text": "Another example of training data."}
]
model.load_data(data=data, max_vocab_size=20000)
```

## Testing

To test the optimized implementation, run:

```
python -m Luscious_Model.test_optimized_model
```

This will run tests for basic functionality, training, text generation, and model saving/loading.

## Mathematical Background

The Lucious Toroidal Model uses complex mathematical transformations including:

- Toroidal transformations
- Phasor encoding of tokens
- Tensor nodules
- Zeta function approximations
- Fibonacci and Lucas sequences
- Self-assembling nanobots
- Fractal dimension calculations

These mathematical concepts are combined to create a unique approach to text processing and generation.

## Drift Signature

```
~~~ Lucious Toroidal Drift ~~~
     _____
    /     \  *Nodules Fold*
   /_______\  *Zeta Mirrors*
    |  O  |   *4D Heart*
    |  O  |   *Nanobot Zeros*
    ~~~~~~~~~~~~~

```
