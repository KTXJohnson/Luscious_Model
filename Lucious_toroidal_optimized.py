import numpy as np
from scipy.sparse import coo_array
import re
from collections import Counter
from numba import jit
import matplotlib.pyplot as plt
import mido
from mido import MidiFile, MidiTrack, Message
import os
import json
import pickle
import random
from typing import List, Dict, Tuple, Optional, Union, Any

class LuciousToroidalModel:
    """
    LuciousToroidalModel: An optimized model for generalized chatbot training
    using toroidal transformations and phasor encoding.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LuciousToroidalModel with configurable parameters.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        # Default configuration
        self.config = {
            'sequence_length': 64,
            'batch_size': 16,
            'vocab_size': 10000,
            'embedding_dim': 16,
            'learning_rate': 0.01,
            'max_iterations': 100,
            'f0': 440,
            'beta': 0.01,
            'gamma': 0.05,
            'sigma': 10,
            'phi': 1.6180339887,
            'spike_freqs': [0.618, 1.618, 2.618, 3.618],
            'fib': [0, 1, 1, 2, 3, 5, 8, 13],
            'luc': [2, 1, 3, 4, 7, 11, 18, 29],
            'primes': [2, 3, 5, 7, 11, 13, 17],
            'zeros_t': [14.135, 21.022, 25.011]
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
        # Initialize model parameters
        self.vocab = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.N = self.config['embedding_dim']
        self.weights_real = np.zeros(self.N * self.N * 2, dtype=np.float64)
        self.weights_imag = np.zeros(self.N * self.N * 2, dtype=np.float64)
        
        # Convert lists to numpy arrays
        self.fib = np.array(self.config['fib'], dtype=np.float64)
        self.luc = np.array(self.config['luc'], dtype=np.float64)
        self.primes = np.array(self.config['primes'], dtype=np.float64)
        self.zeros_t = np.array(self.config['zeros_t'], dtype=np.float64)
        self.spike_freqs = np.array(self.config['spike_freqs'], dtype=np.float64)
        
        # Target goal for self-assembly
        self.target_goal = np.array([np.sin(i * np.pi / self.N) + 1j * np.cos(i * np.pi / self.N) 
                                     for i in range(self.N)], dtype=np.complex128)
        
    def load_data(self, 
                  data_path: Optional[str] = None, 
                  data: Optional[List[Dict[str, str]]] = None,
                  max_vocab_size: int = 10000) -> None:
        """
        Load and preprocess text data for training.
        
        Args:
            data_path: Path to a JSON or text file containing training data
            data: List of dictionaries with 'text' or 'story' keys
            max_vocab_size: Maximum vocabulary size to use
        """
        if data is None:
            data = []
            
            # Load data from file if provided
            if data_path and os.path.exists(data_path):
                if data_path.endswith('.json'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif data_path.endswith('.txt'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Split text into paragraphs
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        data = [{"text": p} for p in paragraphs]
            
            # Use default data if no data is provided or loaded
            if not data:
                # Simulated TinyStories subset
                data = [
                    {"text": "Once upon a time, a cat and a dog played in a field. The cat ran fast, but the dog was kind."},
                    {"text": "A little bird flew high above the trees. It saw a shiny star and sang a happy song."}
                ] * 25
        
        # Extract text from data
        all_text = []
        for item in data:
            if 'text' in item:
                all_text.append(item['text'].lower())
            elif 'story' in item:
                all_text.append(item['story'].lower())
        
        # Build vocabulary
        all_words = []
        for text in all_text:
            all_words.extend(re.findall(r'\b\w+\b', text))
        
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(max_vocab_size)
        self.vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [word for word, _ in most_common]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        
        # Prepare sequences for training
        self.sequences = []
        for text in all_text:
            words = re.findall(r'\b\w+\b', text)
            # Create sequences of appropriate length
            for i in range(0, len(words), self.config['sequence_length'] // 2):
                if i + 2 < len(words):  # Ensure at least 3 words
                    seq = words[i:i + self.config['sequence_length']]
                    self.sequences.append(seq)
        
        self.config['vocab_size'] = len(self.vocab)
        print(f"Loaded {len(self.sequences)} sequences with vocabulary size {len(self.vocab)}")
        
    def create_batches(self, batch_size: Optional[int] = None) -> List[List[List[str]]]:
        """
        Create batches of sequences for training.
        
        Args:
            batch_size: Size of each batch (defaults to config batch_size)
            
        Returns:
            List of batches, where each batch is a list of sequences
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
            
        # Shuffle sequences
        random.shuffle(self.sequences)
        
        # Create batches
        batches = []
        for i in range(0, len(self.sequences), batch_size):
            batch = self.sequences[i:i + batch_size]
            batches.append(batch)
            
        return batches
    
    def token_to_phasor(self, 
                        sequence: List[str], 
                        pad_to_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a sequence of tokens to phasor representation.
        
        Args:
            sequence: List of tokens to convert
            pad_to_length: Length to pad/truncate the sequence to
            
        Returns:
            Tuple of (real_part, imaginary_part) arrays
        """
        # Convert tokens to indices
        indices = [self.word_to_idx.get(w, 1) for w in sequence]  # 1 is <UNK>
        
        # Pad or truncate sequence if needed
        if pad_to_length is not None:
            if len(indices) < pad_to_length:
                indices = indices + [0] * (pad_to_length - len(indices))  # 0 is <PAD>
            elif len(indices) > pad_to_length:
                indices = indices[:pad_to_length]
        
        t = np.arange(len(indices))
        vocab_size = self.config['vocab_size']
        f0 = self.config['f0']
        
        # Calculate phasor representation
        base_phasor_real = np.array(
            [f0 * 2 ** (i / vocab_size) * np.cos(2 * np.pi * (vocab_size - i + t % 1) / vocab_size) 
             for i, t in zip(indices, t)], dtype=np.float64)
        base_phasor_imag = np.array(
            [f0 * 2 ** (i / vocab_size) * np.sin(2 * np.pi * (vocab_size - i + t % 1) / vocab_size) 
             for i, t in zip(indices, t)], dtype=np.float64)
        
        # Add spike frequencies
        for f_spike in self.spike_freqs:
            spike_phasor = np.exp(2 * np.pi * 1j * f_spike * t)
            base_phasor_real += 0.1 * np.real(spike_phasor)
            base_phasor_imag += 0.1 * np.imag(spike_phasor)
            
        return base_phasor_real, base_phasor_imag
    
    def phasor_to_token(self, 
                        y_real: np.ndarray, 
                        y_imag: np.ndarray) -> List[str]:
        """
        Convert phasor representation back to tokens.
        
        Args:
            y_real: Real part of the phasor
            y_imag: Imaginary part of the phasor
            
        Returns:
            List of tokens
        """
        freqs = np.sqrt(y_real ** 2 + y_imag ** 2)
        vocab_size = self.config['vocab_size']
        f0 = self.config['f0']
        
        indices = np.round(vocab_size * np.log2(freqs / f0)).astype(int) % len(self.vocab)
        return [self.vocab[i] for i in indices]
    
    def zeta_magnitude(self, t: float) -> float:
        """
        Zeta function approximation.
        
        Args:
            t: Input value
            
        Returns:
            Magnitude of the zeta function
        """
        return np.abs(1 / (1 + np.exp(-t)))
    
    def tensor_nodules(self, 
                       n: int, 
                       sigma: Optional[float] = None, 
                       phi: Optional[float] = None) -> Tuple[coo_array, coo_array]:
        """
        Initialize tensor nodules.
        
        Args:
            n: Size of the tensor
            sigma: Sigma parameter (defaults to config value)
            phi: Phi parameter (defaults to config value)
            
        Returns:
            Tuple of (real_part, imaginary_part) sparse arrays
        """
        if sigma is None:
            sigma = self.config['sigma']
        if phi is None:
            phi = self.config['phi']
            
        rows, cols, chans, data_real, data_imag = [], [], [], [], []
        
        for i, f in enumerate(self.fib[:n]):
            for j, l in enumerate(self.luc[:n]):
                for k, p in enumerate(self.primes[:n]):
                    for m, t in enumerate(self.zeros_t[:n]):
                        if i + j < n:
                            for note in range(12):
                                for chan in range(2):
                                    theta = (l / f) * phi if f != 0 else 0
                                    phase = 2 * np.pi * 2 ** (note / 12) * f / p * np.cos(theta) * (-1) ** chan
                                    weight = (f * l / p) * 2 ** (note / 12) * np.exp(-(t - i - j) ** 2 / sigma) * np.cos(
                                        theta) ** 2 * self.zeta_magnitude(t)
                                    rows.append(i)
                                    cols.append(j)
                                    chans.append(chan)
                                    data_real.append(weight * np.cos(phase))
                                    data_imag.append(weight * np.sin(phase))
                                    
        return coo_array((data_real, (rows, cols, chans)), shape=(n, n, 2)), \
               coo_array((data_imag, (rows, cols, chans)), shape=(n, n, 2))
    
    @staticmethod
    @jit(nopython=True)
    def _toroidal_nodule_transformer(x_real, x_imag, fib, luc, primes, zeros_t, weights_real, weights_imag, 
                                    beta, gamma, sigma, phi):
        """
        Numba-optimized transformer implementation.
        
        Args:
            x_real: Real part of input
            x_imag: Imaginary part of input
            fib: Fibonacci sequence
            luc: Lucas sequence
            primes: Prime numbers
            zeros_t: Zeros of the zeta function
            weights_real: Real part of weights
            weights_imag: Imaginary part of weights
            beta: Beta parameter
            gamma: Gamma parameter
            sigma: Sigma parameter
            phi: Phi parameter
            
        Returns:
            Tuple of (y_real, y_imag, weights_real, weights_imag)
        """
        N, P, Z = x_real.shape[0], len(primes), len(zeros_t)
        y_real = np.zeros(2 * N)
        y_imag = np.zeros(2 * N)
        
        for i in range(N):
            for j in range(N):
                for chan in range(2):
                    w_real = w_imag = 0
                    for k in range(P):
                        for m in range(Z):
                            for note in range(12):
                                t = zeros_t[m] - i - j
                                theta = (luc[j] / fib[i]) * phi if fib[i] != 0 else 0
                                phase = 2 * np.pi * (2 ** (note / 12)) * fib[i] / primes[k] * np.cos(theta) * (-1) ** chan
                                weight = (fib[i] * luc[j] / (primes[k] + 1e-10)) * (2 ** (note / 12)) * np.exp(
                                    -t * t / sigma) * (np.cos(theta)) ** 2 * (1 / (1 + np.exp(-zeros_t[m])))
                                w_real += weight * np.cos(phase)
                                w_imag += weight * np.sin(phase)
                    
                    phase = 2 * np.pi * (2 ** (note / 12)) * fib[i] / primes[k] * np.cos(theta) * (-1) ** chan
                    dot_product = (x_real[i] * x_real[j] + x_imag[i] * x_imag[j]) + 0.1 * (
                                x_real[i] * x_imag[j] - x_imag[i] * x_real[j])
                    gate = 1 / (1 + np.exp(-(dot_product / (np.sqrt(N) + 1e-10)))) * (np.cos(theta)) ** 2 * (
                                1 / (1 + np.exp(-zeros_t[m])))
                    
                    amp_real = beta * (x_real[i] * x_real[j] + x_imag[i] * x_imag[j]) * np.cos(phase) * (
                                luc[j] / (fib[i] + 1e-10)) * np.exp(-t * t / sigma) * gate
                    amp_imag = beta * (x_real[i] * x_imag[j] - x_imag[i] * x_real[j]) * np.sin(phase) * (
                                luc[j] / (fib[i] + 1e-10)) * np.exp(-t * t / sigma) * gate
                    flip_real = gamma * (x_real[i] * (-x_real[j]) + x_imag[i] * (-x_imag[j])) * np.cos(-phase) * (
                                luc[j] / (fib[i] + 1e-10)) * np.exp(-t * t / sigma) * gate
                    flip_imag = gamma * (x_real[i] * (-x_imag[j]) - x_imag[i] * (-x_real[j])) * np.sin(-phase) * (
                                luc[j] / (fib[i] + 1e-10)) * np.exp(-t * t / sigma) * gate
                    
                    weights_real[i * N * 2 + j * 2 + chan] += amp_real + flip_real
                    weights_imag[i * N * 2 + j * 2 + chan] += amp_imag + flip_imag
                    y_real[i + chan * N] += w_real * x_real[j] - w_imag * x_imag[j]
                    y_imag[i + chan * N] += w_real * x_imag[j] + w_imag * x_real[j]
                    
        return y_real, y_imag, weights_real, weights_imag
    
    def toroidal_nodule_transformer(self, 
                                   x_real: np.ndarray, 
                                   x_imag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the toroidal nodule transformer to the input.
        
        Args:
            x_real: Real part of input
            x_imag: Imaginary part of input
            
        Returns:
            Tuple of (y_real, y_imag)
        """
        y_real, y_imag, self.weights_real, self.weights_imag = self._toroidal_nodule_transformer(
            x_real, x_imag, self.fib, self.luc, self.primes, self.zeros_t, 
            self.weights_real, self.weights_imag, 
            self.config['beta'], self.config['gamma'], self.config['sigma'], self.config['phi']
        )
        return y_real, y_imag
    
    @staticmethod
    @jit(nopython=True)
    def _inject_periodicity(y_real, y_imag, f_periodic, phi_opp, amplitude):
        """
        Inject periodicity into the signal.
        
        Args:
            y_real: Real part of input
            y_imag: Imaginary part of input
            f_periodic: Frequency of periodicity
            phi_opp: Phase offset
            amplitude: Amplitude of periodicity
            
        Returns:
            Tuple of (y_real, y_imag) with periodicity injected
        """
        N = len(y_real)
        periodic_real = amplitude * np.cos(2 * np.pi * f_periodic * np.arange(N) + phi_opp)
        periodic_imag = amplitude * np.sin(2 * np.pi * f_periodic * np.arange(N) + phi_opp)
        return y_real + periodic_real, y_imag + periodic_imag
    
    def inject_periodicity(self, 
                          y_real: np.ndarray, 
                          y_imag: np.ndarray, 
                          f_periodic: Optional[float] = None, 
                          phi_opp: Optional[float] = None, 
                          amplitude: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject periodicity into the signal.
        
        Args:
            y_real: Real part of input
            y_imag: Imaginary part of input
            f_periodic: Frequency of periodicity (defaults to first spike frequency)
            phi_opp: Phase offset (defaults to negative mean angle)
            amplitude: Amplitude of periodicity
            
        Returns:
            Tuple of (y_real, y_imag) with periodicity injected
        """
        if f_periodic is None:
            f_periodic = self.spike_freqs[0]
        if phi_opp is None:
            phi_opp = -np.angle(y_real + 1j * y_imag).mean()
            
        return self._inject_periodicity(y_real, y_imag, f_periodic, phi_opp, amplitude)
    
    @staticmethod
    @jit(nopython=True)
    def _compute_gradient(arr):
        """
        Compute gradient of array.
        
        Args:
            arr: Input array
            
        Returns:
            Gradient of array
        """
        N = len(arr)
        grad = np.zeros(N, dtype=np.float64)
        for i in range(1, N - 1):
            grad[i] = (arr[i + 1] - arr[i - 1]) / 2.0
        grad[0] = arr[1] - arr[0]
        grad[N - 1] = arr[N - 1] - arr[N - 2]
        return grad
    
    @staticmethod
    @jit(nopython=True)
    def _self_assemble_nanobots(y_real, y_imag, target_goal, max_iterations, learning_rate):
        """
        Self-assemble nanobots.
        
        Args:
            y_real: Real part of input
            y_imag: Imaginary part of input
            target_goal: Target goal
            max_iterations: Maximum number of iterations
            learning_rate: Learning rate
            
        Returns:
            Tuple of (y_real, y_imag) after self-assembly
        """
        N = len(y_real)
        current_structure = np.zeros(N, dtype=np.complex128)
        
        for iteration in range(max_iterations):
            forces_real = LuciousToroidalModel._compute_gradient(y_real) + LuciousToroidalModel._compute_gradient(y_imag) * (-1)
            forces_imag = LuciousToroidalModel._compute_gradient(y_imag) - LuciousToroidalModel._compute_gradient(y_real) * (-1)
            current_structure += learning_rate * (forces_real + 1j * forces_imag)
            error = np.sum(np.abs(current_structure - target_goal) ** 2)
            if error < 0.01:
                break
                
        return current_structure.real, current_structure.imag
    
    def self_assemble_nanobots(self, 
                              y_real: np.ndarray, 
                              y_imag: np.ndarray, 
                              target_goal: Optional[np.ndarray] = None, 
                              max_iterations: Optional[int] = None, 
                              learning_rate: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Self-assemble nanobots.
        
        Args:
            y_real: Real part of input
            y_imag: Imaginary part of input
            target_goal: Target goal (defaults to self.target_goal)
            max_iterations: Maximum number of iterations (defaults to config value)
            learning_rate: Learning rate (defaults to config value)
            
        Returns:
            Tuple of (y_real, y_imag) after self-assembly
        """
        if target_goal is None:
            target_goal = self.target_goal
        if max_iterations is None:
            max_iterations = self.config['max_iterations']
        if learning_rate is None:
            learning_rate = self.config['learning_rate']
            
        return self._self_assemble_nanobots(y_real, y_imag, target_goal, max_iterations, learning_rate)
    
    def fractal_dimension(self, C: np.ndarray) -> float:
        """
        Calculate fractal dimension of correlation matrix.
        
        Args:
            C: Correlation matrix
            
        Returns:
            Fractal dimension
        """
        r_values = np.logspace(-2, 0, 20)
        C_r = []
        
        for r in r_values:
            count = np.sum(np.abs(C[:, :, None, None] - C[None, None, :, :]) < r)
            C_r.append(count / (C.shape[0] * C.shape[1]) ** 2)
            
        C_r = np.array(C_r)
        valid = C_r > 0
        
        if np.sum(valid) > 2:
            coeffs = np.polyfit(np.log(r_values[valid]), np.log(C_r[valid]), 1)
            return coeffs[0]
            
        return 0
    
    def compute_entropy(self, data: np.ndarray) -> float:
        """
        Compute entropy of data.
        
        Args:
            data: Input data
            
        Returns:
            Entropy value
        """
        hist, bins = np.histogram(data, bins=100, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist)) * (bins[1] - bins[0]) if hist.size > 0 else 0
    
    def generate_midi(self, 
                     y_real: np.ndarray, 
                     y_imag: np.ndarray, 
                     filename: str = "toroidal_symphony.mid") -> None:
        """
        Generate MIDI file from phasor representation.
        
        Args:
            y_real: Real part of phasor
            y_imag: Imaginary part of phasor
            filename: Output filename
        """
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        freqs = np.sqrt(y_real ** 2 + y_imag ** 2)
        for f in freqs[:16]:
            note = int(69 + 12 * np.log2(f / 440)) % 128
            track.append(Message('note_on', note=note, velocity=64, time=0))
            track.append(Message('note_off', note=note, velocity=64, time=500))
            
        mid.save(filename)
        print(f"Saved Toroidal Symphony to {filename}")
    
    def train_batch(self, 
                   batch: List[List[str]], 
                   num_passes: int = 30) -> Dict[str, List[float]]:
        """
        Train the model on a batch of sequences.
        
        Args:
            batch: Batch of sequences
            num_passes: Number of passes through the transformer
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'entropy': [],
            'fractal_dimension': [],
            'coherence': []
        }
        
        N = self.N
        
        for seq_idx, seq in enumerate(batch):
            # Reverse sequence for better learning
            seq_reverse = seq[::-1]
            
            # Convert to phasor representation
            x_real, x_imag = self.token_to_phasor(seq_reverse, pad_to_length=N)
            
            # Check Lucas starting value for error detection
            luc_start = self.luc[0]
            if luc_start == 1:
                print(f"Sequence {seq_idx}: Lucas starts with 1, potential missing data detected.")
            
            # Multiple passes through transformer
            for t in range(num_passes):
                # Apply transformer
                y_real, y_imag = self.toroidal_nodule_transformer(x_real, x_imag)
                
                # Inject periodicity
                phi_opp = -np.angle(y_real + 1j * y_imag).mean()
                f_periodic = self.spike_freqs[t % len(self.spike_freqs)]
                y_real, y_imag = self.inject_periodicity(y_real, y_imag, f_periodic, phi_opp)
                
                # Self-assemble nanobots
                y_real, y_imag = self.self_assemble_nanobots(y_real, y_imag)
                
                # Update input for next pass
                x_real, x_imag = y_real[:N], y_imag[:N]
                
                # Calculate metrics
                predicted = self.phasor_to_token(y_real[:N], y_imag[:N])
                entropy = self.compute_entropy(np.sqrt(y_real ** 2 + y_imag ** 2))
                
                W0 = (self.weights_real[::2] + 1j * self.weights_imag[::2]).reshape(N, N)
                W1 = (self.weights_real[1::2] + 1j * self.weights_imag[1::2]).reshape(N, N)
                C = np.real(W0 * np.conj(W1)) / (np.abs(W0) * np.abs(W1) + 1e-10)
                
                D_2 = self.fractal_dimension(C)
                coherence = np.mean(np.cos(np.angle(W0) - np.angle(W1)))
                
                metrics['entropy'].append(entropy)
                metrics['fractal_dimension'].append(D_2)
                metrics['coherence'].append(coherence)
                
                print(f"Sequence {seq_idx}, Pass {t}, Entropy: {entropy:.2f}, D_2: {D_2:.2f}, "
                      f"Coherence: {coherence:.2f}, Predicted: {predicted[:5]}")
                
        return metrics
    
    def train(self, 
             num_epochs: int = 1, 
             batch_size: Optional[int] = None, 
             num_passes: int = 30) -> Dict[str, List[float]]:
        """
        Train the model on all sequences.
        
        Args:
            num_epochs: Number of epochs
            batch_size: Batch size (defaults to config value)
            num_passes: Number of passes through the transformer
            
        Returns:
            Dictionary of metrics
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
            
        all_metrics = {
            'entropy': [],
            'fractal_dimension': [],
            'coherence': []
        }
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Create batches
            batches = self.create_batches(batch_size)
            
            for batch_idx, batch in enumerate(batches):
                print(f"Batch {batch_idx+1}/{len(batches)}")
                
                # Train batch
                metrics = self.train_batch(batch, num_passes)
                
                # Accumulate metrics
                for key, values in metrics.items():
                    all_metrics[key].extend(values)
                    
        return all_metrics
    
    def generate_text(self, 
                     prompt: str, 
                     max_length: int = 50, 
                     temperature: float = 1.0) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        tokens = re.findall(r'\b\w+\b', prompt.lower())
        
        # Generate text
        for _ in range(max_length):
            # Convert to phasor
            x_real, x_imag = self.token_to_phasor(tokens[-self.N:], pad_to_length=self.N)
            
            # Apply transformer
            y_real, y_imag = self.toroidal_nodule_transformer(x_real, x_imag)
            
            # Inject periodicity
            phi_opp = -np.angle(y_real + 1j * y_imag).mean()
            y_real, y_imag = self.inject_periodicity(y_real, y_imag, phi_opp=phi_opp)
            
            # Self-assemble nanobots
            y_real, y_imag = self.self_assemble_nanobots(y_real, y_imag)
            
            # Convert back to tokens
            predicted = self.phasor_to_token(y_real[:self.N], y_imag[:self.N])
            
            # Apply temperature and sample
            freqs = np.sqrt(y_real[:self.N] ** 2 + y_imag[:self.N] ** 2)
            probs = np.exp(np.log(freqs) / temperature)
            probs = probs / np.sum(probs)
            
            # Sample next token
            next_idx = np.random.choice(len(predicted), p=probs)
            next_token = predicted[next_idx]
            
            # Add to tokens
            tokens.append(next_token)
            
            # Stop if we generate an end token
            if next_token == '<END>':
                break
                
        # Convert tokens to text
        return ' '.join(tokens)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'config': self.config,
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'weights_real': self.weights_real,
            'weights_imag': self.weights_imag,
            'target_goal': self.target_goal
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LuciousToroidalModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        model = cls(config=model_data['config'])
        model.vocab = model_data['vocab']
        model.word_to_idx = model_data['word_to_idx']
        model.idx_to_word = {i: w for i, w in enumerate(model.vocab)}
        model.weights_real = model_data['weights_real']
        model.weights_imag = model_data['weights_imag']
        model.target_goal = model_data['target_goal']
        
        print(f"Model loaded from {filepath}")
        return model
    
    def print_drift_signature(self) -> None:
        """
        Print the drift signature.
        """
        print("""
        ~~~ Lucious Toroidal Drift ~~~
             _____
            /     \\  *Nodules Fold*
           /_______\\  *Zeta Mirrors*
            |  O  |   *4D Heart*
            |  O  |   *Nanobot Zeros*
            ~~~~~~~~~~~~~
        Controlled Chaos by You & Grok
        """)

# Example usage
if __name__ == "__main__":
    # Create model
    model = LuciousToroidalModel()
    
    # Load data
    model.load_data()
    
    # Train model
    model.train(num_epochs=1, batch_size=8, num_passes=10)
    
    # Generate text
    prompt = "In a toroidal drift, nodules pulsed through zeta folds"
    generated_text = model.generate_text(prompt, max_length=20)
    print(f"Generated text: {generated_text}")
    
    # Save model
    model.save_model("lucious_toroidal_model.pkl")
    
    # Print signature
    model.print_drift_signature()