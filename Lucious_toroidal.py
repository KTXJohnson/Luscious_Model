import numpy as np
from scipy.sparse import coo_array
import re
from collections import Counter
from numba import jit
import matplotlib.pyplot as plt
import mido
from mido import MidiFile, MidiTrack, Message

# Simulated TinyStories subset
subset_stories = [
                     {
                         "story": "Once upon a time, a cat and a dog played in a field. The cat ran fast, but the dog was kind."},
                     {"story": "A little bird flew high above the trees. It saw a shiny star and sang a happy song."},
                 ] * 25
vocab = set(word for story in subset_stories for word in re.findall(r'\b\w+\b', story["story"].lower()))
vocab = list(vocab)[:3000]
word_to_idx = {w: i for i, w in enumerate(vocab)}
sequences = [re.findall(r'\b\w+\b', s["story"].lower())[:10] for s in subset_stories]

# Toroidal zeta prompts
prompts = [
    "In a toroidal drift, nodules pulsed through zeta folds, weaving antimatter mirrors.",
    "O’er a 4D grid, Yin and Yang spun backward, unifying cosmic zeros.",
    "A starry cat and dog pulsed, their spins folding space in zeta’s dance."
]


# Token to 4D phasor with golden ratio spikes
def token_to_phasor(sequence, vocab_size, word_to_idx, f0=440, spike_freqs=None):
    indices = [word_to_idx.get(w, 0) for w in sequence]
    t = np.arange(len(indices))
    base_phasor_real = np.array(
        [f0 * 2 ** (i / vocab_size) * np.cos(2 * np.pi * (vocab_size - i + t % 1 + t % 1 + t % 1) / vocab_size) for i, t
         in zip(indices, t)], dtype=np.float64)
    base_phasor_imag = np.array(
        [f0 * 2 ** (i / vocab_size) * np.sin(2 * np.pi * (vocab_size - i + t % 1 + t % 1 + t % 1) / vocab_size) for i, t
         in zip(indices, t)], dtype=np.float64)
    if spike_freqs is not None:
        for f_spike in spike_freqs:
            spike_phasor = np.exp(2 * np.pi * 1j * f_spike * t)
            base_phasor_real += 0.1 * np.real(spike_phasor)
            base_phasor_imag += 0.1 * np.imag(spike_phasor)
    return base_phasor_real, base_phasor_imag


# Phasor to token
def phasor_to_token(y_real, y_imag, vocab, f0=440):
    freqs = np.sqrt(y_real ** 2 + y_imag ** 2)
    indices = np.round(vocab_size * np.log2(freqs / f0)).astype(int) % len(vocab)
    return [vocab[i] for i in indices]


# Zeta function approximation
def zeta_magnitude(t):
    return np.abs(1 / (1 + np.exp(-t)))


# Tensor Nodule initializer
def tensor_nodules(n, fib, luc, primes, zeros_t, sigma=10, phi=1.6180339887):
    rows, cols, chans, data_real, data_imag = [], [], [], [], []
    for i, f in enumerate(fib[:n]):
        for j, l in enumerate(luc[:n]):
            for k, p in enumerate(primes[:n]):
                for m, t in enumerate(zeros_t[:n]):
                    if i + j < n:
                        for note in range(12):
                            for chan in range(2):
                                theta = (l / f) * phi if f != 0 else 0
                                phase = 2 * np.pi * 2 ** (note / 12) * f / p * np.cos(theta) * (-1) ** chan
                                weight = (f * l / p) * 2 ** (note / 12) * np.exp(-(t - i - j) ** 2 / sigma) * np.cos(
                                    theta) ** 2 * zeta_magnitude(t)
                                rows.append(i)
                                cols.append(j)
                                chans.append(chan)
                                data_real.append(weight * np.cos(phase))
                                data_imag.append(weight * np.sin(phase))
    return coo_array((data_real, (rows, cols, chans)), shape=(n, n, 2)), \
        coo_array((data_imag, (rows, cols, chans)), shape=(n, n, 2))


# Numba-optimized transformer
@jit(nopython=True)
def toroidal_nodule_transformer(x_real, x_imag, fib, luc, primes, zeros_t, weights_real, weights_imag, beta=0.01,
                                gamma=0.05, sigma=10, phi=1.6180339887):
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


# Periodicity injection for nanobots
@jit(nopython=True)
def inject_periodicity(y_real, y_imag, f_periodic=1.0, phi_opp=0.0, amplitude=0.1):
    N = len(y_real)
    periodic_real = amplitude * np.cos(2 * np.pi * f_periodic * np.arange(N) + phi_opp)
    periodic_imag = amplitude * np.sin(2 * np.pi * f_periodic * np.arange(N) + phi_opp)
    return y_real + periodic_real, y_imag + periodic_imag


# Custom gradient for Numba
@jit(nopython=True)
def compute_gradient(arr):
    N = len(arr)
    grad = np.zeros(N, dtype=np.float64)
    for i in range(1, N - 1):
        grad[i] = (arr[i + 1] - arr[i - 1]) / 2.0
    grad[0] = arr[1] - arr[0]
    grad[N - 1] = arr[N - 1] - arr[N - 2]
    return grad


# Nanobot self-assembly
@jit(nopython=True)
def self_assemble_nanobots(y_real, y_imag, target_goal, max_iterations=100, learning_rate=0.01):
    N = len(y_real)
    current_structure = np.zeros(N, dtype=np.complex128)
    for iteration in range(max_iterations):
        forces_real = compute_gradient(y_real) + compute_gradient(y_imag) * (-1)
        forces_imag = compute_gradient(y_imag) - compute_gradient(y_real) * (-1)
        current_structure += learning_rate * (forces_real + 1j * forces_imag)
        error = np.sum(np.abs(current_structure - target_goal) ** 2)
        if error < 0.01:
            break
    return current_structure.real, current_structure.imag


# Fractal dimension
def fractal_dimension(C):
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


# MIDI generation
def generate_midi(y_real, y_imag, filename="toroidal_symphony.mid"):
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


# Compute entropy
def compute_entropy(data):
    hist, bins = np.histogram(data, bins=100, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) * (bins[1] - bins[0]) if hist.size > 0 else 0


# Drift signature
def print_drift_signature():
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


# Training loop
fib = np.array([0, 1, 1, 2, 3, 5, 8, 13], dtype=np.float64)
luc = np.array([2, 1, 3, 4, 7, 11, 18, 29], dtype=np.float64)  # Start with 2 for success
primes = np.array([2, 3, 5, 7, 11, 13, 17], dtype=np.float64)
zeros_t = np.array([14.135, 21.022, 25.011], dtype=np.float64)
spike_freqs = np.array([0.618, 1.618, 2.618, 3.618], dtype=np.float64)  # Golden ratio spikes
N = 16
vocab_size = len(vocab)
weights_real = np.zeros(N * N * 2, dtype=np.float64)
weights_imag = np.zeros(N * N * 2, dtype=np.float64)

# Simulated target goal (e.g., 3D dome coordinates)
target_goal = np.array([np.sin(i * np.pi / N) + 1j * np.cos(i * np.pi / N) for i in range(N)], dtype=np.complex128)

# Train with prompts
for seq_idx, seq in enumerate(sequences[:50]):
    if seq_idx % 10 == 0:
        seq = prompts[seq_idx % len(prompts)].split()[:10]
    seq_reverse = seq[::-1]
    x_real, x_imag = token_to_phasor(seq_reverse, vocab_size, word_to_idx, spike_freqs=spike_freqs)

    # Check Lucas starting value for error detection
    luc_start = luc[0]
    if luc_start == 1:
        print(f"Sequence {seq_idx}: Lucas starts with 1, potential missing data detected.")

    for t in range(30):
        y_real, y_imag, weights_real, weights_imag = toroidal_nodule_transformer(
            x_real, x_imag, fib, luc, primes, zeros_t, weights_real, weights_imag
        )
        # Inject periodicity for nanobots
        phi_opp = -np.angle(y_real + 1j * y_imag).mean()  # Opposing phase
        y_real, y_imag = inject_periodicity(y_real, y_imag, f_periodic=spike_freqs[t % len(spike_freqs)],
                                            phi_opp=phi_opp)
        # Self-assemble nanobots
        y_real, y_imag = self_assemble_nanobots(y_real, y_imag, target_goal)
        x_real, x_imag = y_real[:N], y_imag[:N]
        predicted = phasor_to_token(y_real[:N], y_imag[:N], vocab)
        entropy = compute_entropy(np.sqrt(y_real ** 2 + y_imag ** 2))
        W0 = (weights_real[::2] + 1j * weights_imag[::2]).reshape(N, N)
        W1 = (weights_real[1::2] + 1j * weights_imag[1::2]).reshape(N, N)
        C = np.real(W0 * np.conj(W1)) / (np.abs(W0) * np.abs(W1) + 1e-10)
        D_2 = fractal_dimension(C)
        coherence = np.mean(np.cos(np.angle(W0) - np.angle(W1)))
        print(
            f"Sequence {seq_idx}, Pass {t}, Entropy: {entropy:.2f}, D_2: {D_2:.2f}, Coherence: {coherence:.2f}, Predicted: {predicted[:5]}")

# Lucas-scaled output
lucas_weights = np.array([w * l for w, l in zip(weights_real, luc[:len(weights_real) // len(luc) + 1])])
prompt = prompts[0].split()[:10]
x_real, x_imag = token_to_phasor(prompt[::-1], vocab_size, word_to_idx, spike_freqs=spike_freqs)
y_real, y_imag, _, _ = toroidal_nodule_transformer(x_real, x_imag, fib, luc, primes, zeros_t, lucas_weights,
                                                   weights_imag)
y_real, y_imag = inject_periodicity(y_real, y_imag, f_periodic=spike_freqs[0],
                                    phi_opp=-np.angle(y_real + 1j * y_imag).mean())
y_real, y_imag = self_assemble_nanobots(y_real, y_imag, target_goal)
print(f"Toroidal Nodule Output: {phasor_to_token(y_real[:N], y_imag[:N], vocab)}")
generate_midi(y_real, y_imag)
print_drift_signature()

# Visualize spins
plt.figure(figsize=(8, 8))
plt.imshow(C, cmap='twilight', extent=[0, N, 0, N])
plt.title("Toroidal Nodule Spin Correlations")
plt.colorbar(label="C_ij")
plt.savefig("toroidal_spins.png")
plt.close()
print("Saved toroidal spin visualization to toroidal_spins.png")