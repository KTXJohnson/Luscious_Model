import matplotlib.pyplot as plt
import mido
from mido import MidiFile, MidiTrack, Message

# Simulated TinyStories subset
subset_stories = [
    {"story": "Once upon a time, a cat and a dog played in a field. The cat ran fast, but the dog was kind."},
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
def token_to_phasor(sequence, vocab_size, word_to_idx, f0=440, spike_freqs=None, N=16):
    indices = [word_to_idx.get(w, 0) for w in sequence]
    t = np.arange(len(indices))
    base_phasor_real = np.array([f0 * 2**(i/vocab_size) * np.cos(2 * np.pi * (vocab_size - i + t % 1) / vocab_size) for i, t in zip(indices, t)], dtype=np.float64)
    base_phasor_imag = np.array([f0 * 2**(i/vocab_size) * np.sin(2 * np.pi * (vocab_size - i + t % 1) / vocab_size) for i, t in zip(indices, t)], dtype=np.float64)
    if spike_freqs is not None:
        for f_spike in spike_freqs:
            spike_phasor = np.exp(2 * np.pi * 1j * f_spike * t)
            base_phasor_real += 0.1 * np.real(spike_phasor)
            base_phasor_imag += 0.1 * np.imag(spike_phasor)
    if len(base_phasor_real) < N:
        base_phasor_real = np.pad(base_phasor_real, (0, N - len(base_phasor_real)), mode='constant')
        base_phasor_imag = np.pad(base_phasor_imag, (0, N - len(base_phasor_imag)), mode='constant')
    elif len(base_phasor_real) > N:
        base_phasor_real = base_phasor_real[:N]
        base_phasor_imag = base_phasor_imag[:N]
    return base_phasor_real, base_phasor_imag

# Phasor to token
def phasor_to_token(y_real, y_imag, vocab, f0=440):
    freqs = np.sqrt(y_real**2 + y_imag**2)
    indices = np.round(vocab_size * np.log2(freqs / f0)).astype(int) % len(vocab)
    return [vocab[i] for i in indices]

# Zeta function approximation
def zeta_magnitude(t):
    return np.abs(1 / (1 + np.exp(-t)))

# Antimatter mirror for inversion
@jit(nopython=True)
def mirror_antimatter(y_real, y_imag, luc, t, phi=1.6180339887):
    N = len(y_real)
    theta = luc[0] * phi if luc[0] != 0 else 0
    scale = zeta_magnitude(t)
    mirrored_real = -y_real * np.cos(theta) * scale
    mirrored_imag = -y_imag * np.sin(theta) * scale
    return mirrored_real, mirrored_imag

# Gated attention mechanism
@jit(nopython=True)
def gated_attention(y_real, y_imag, weights, fib, luc, t, N, d_k=16, beta=0.01):
    Q = y_real * weights[0]
    K = y_imag * weights[1]
    V = y_real * weights[2]
    theta = (luc[0] / (fib[0] + 1e-10)) * 1.6180339887
    G = 1 / (1 + np.exp(-zeta_magnitude(t))) * np.cos(theta)**2
    Q_gated = G * Q
    scores = np.dot(Q_gated, K.T) / np.sqrt(d_k)
    softmax = np.exp(scores) / (np.sum(np.exp(scores), axis=-1, keepdims=True) + 1e-10)
    attention = np.dot(softmax, V)
    y_real_out = beta * attention * np.cos(theta)
    y_imag_out = beta * attention * np.sin(theta)
    return y_real_out, y_imag_out

# Central attention (input, splits into Fibonacci/Lucas)
def central_attention(x_real, x_imag, fib, luc, zeros_t, N, weights):
    y_real, y_imag = x_real.copy(), x_imag.copy()
    t = zeros_t[0]
    y_real, y_imag = gated_attention(y_real, y_imag, weights, fib, luc, t, N)
    y_real_fib = y_real * fib[0] / (fib[0] + 1e-10)
    y_imag_fib = y_imag * fib[0] / (fib[0] + 1e-10)
    y_real_luc = y_real * luc[0] / (luc[0] + 1e-10)
    y_imag_luc = y_imag * luc[0] / (luc[0] + 1e-10)
    return y_real_fib, y_imag_fib, y_real_luc, y_imag_luc

# Output attention (Fibonacci or Lucas)
def output_attention(y_real, y_imag, fib, luc, zeros_t, N, weights, is_fib=True):
    t = zeros_t[1]
    weights_scaled = weights * (fib[0] if is_fib else luc[0]) / (fib[0] + 1e-10)
    y_real_out, y_imag_out = gated_attention(y_real, y_imag, weights_scaled, fib, luc, t, N)
    return y_real_out, y_imag_out

# Singularity collapse
@jit(nopython=True)
def collapse_to_singularity(y_real, y_imag, weights, t, is_fib=True):
    scale = zeta_magnitude(t)
    collapse_real = np.mean(y_real) * scale * weights[0]
    collapse_imag = np.mean(y_imag) * scale * weights[0]
    return collapse_real, collapse_imag

# MIDI generation
def generate_midi(y_real, y_imag, filename="toroidal_symphony.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    freqs = np.sqrt(y_real**2 + y_imag**2)
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

# Fractal dimension
def fractal_dimension(C):
    r_values = np.logspace(-2, 0, 20)
    C_r = []
    for r in r_values:
        count = np.sum(np.abs(C[:, :, None, None] - C[None, None, :, :]) < r)
        C_r.append(count / (C.shape[0] * C.shape[1])**2)
    C_r = np.array(C_r)
    valid = C_r > 0
    if np.sum(valid) > 2:
        coeffs = np.polyfit(np.log(r_values[valid]), np.log(C_r[valid]), 1)
        return coeffs[0]
    return 0

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

# Setup
fib = np.array([0, 1, 1, 2, 3, 5, 8, 13], dtype=np.float64)
luc = np.array([2, 1, 3, 4, 7, 11, 18, 29], dtype=np.float64)
primes = np.array([2, 3, 5, 7, 11, 13, 17], dtype=np.float64)
zeros_t = np.array([14.135, 21.022, 25.011], dtype=np.float64)
spike_freqs = np.array([0.618, 1.618, 2.618, 3.618], dtype=np.float64)
N = 16
vocab_size = len(vocab)
weights = np.random.randn(3, N, N).astype(np.float64)  # Q, K, V weights

# Training loop
for seq_idx, seq in enumerate(sequences[:50]):
    if seq_idx % 10 == 0:
        seq = prompts[seq_idx % len(prompts)].split()[:10]
    seq_reverse = seq[::-1]
    x_real, x_imag = token_to_phasor(seq_reverse, vocab_size, word_to_idx, spike_freqs=spike_freqs, N=N)

    # Central attention (input)
    y_real_fib, y_imag_fib, y_real_luc, y_imag_luc = central_attention(x_real, x_imag, fib, luc, zeros_t, N, weights)

    # Antimatter inversion
    y_real_fib, y_imag_fib = mirror_antimatter(y_real_fib, y_imag_fib, luc, zeros_t[0])
    y_real_luc, y_imag_luc = mirror_antimatter(y_real_luc, y_imag_luc, luc, zeros_t[0])

    # Singularities
    y_real_fib, y_imag_fib = collapse_to_singularity(y_real_fib, y_imag_fib, fib, zeros_t[1], is_fib=True)
    y_real_luc, y_imag_luc = collapse_to_singularity(y_real_luc, y_imag_luc, luc, zeros_t[1], is_fib=False)

    # Output attentions
    y_real_fib, y_imag_fib = output_attention(y_real_fib, y_imag_fib, fib, luc, zeros_t, N, weights, is_fib=True)
    y_real_luc, y_imag_luc = output_attention(y_real_luc, y_imag_luc, fib, luc, zeros_t, N, weights, is_fib=False)

    # Fuse outputs
    y_real = y_real_fib + y_real_luc
    y_imag = y_imag_fib + y_imag_luc

    # Metrics
    predicted = phasor_to_token(y_real, y_imag, vocab)
    entropy = compute_entropy(np.sqrt(y_real**2 + y_imag**2))
    W0 = (y_real_fib + 1j * y_imag_fib).reshape(N, N)
    W1 = (y_real_luc + 1j * y_imag_luc).reshape(N, N)
    C = np.real(W0 * np.conj(W1)) / (np.abs(W0) * np.abs(W1) + 1e-10)
    D_2 = fractal_dimension(C)
    coherence = np.mean(np.cos(np.angle(W0) - np.angle(W1)))
    print(f"Sequence {seq_idx}, Entropy: {entropy:.2f}, D_2: {D_2:.2f}, Coherence: {coherence:.2f}, Predicted: {predicted[:5]}")

# Final output
prompt = prompts[0].split()[:10]
x_real, x_imag = token_to_phasor(prompt[::-1], vocab_size, word_to_idx, spike_freqs=spike_freqs, N=N)
y_real_fib, y_imag_fib, y_real_luc, y_imag_luc = central_attention(x_real, x_imag, fib, luc, zeros_t, N, weights)
y_real_fib, y_imag_fib = mirror_antimatter(y_real_fib, y_imag_fib, luc, zeros_t[0])
y_real_luc, y_imag_luc = mirror_antimatter(y_real_luc, y_imag_luc, luc, zeros_t[0])
y_real_fib, y_imag_fib = collapse_to_singularity(y_real_fib, y_imag_fib, fib, zeros_t[1], is_fib=True)
y_real_luc, y_imag_luc = collapse_to_singularity(y_real_luc, y_imag_luc, luc, zeros_t[1], is_fib=False)
y_real_fib, y_imag_fib = output_attention(y_real_fib, y_imag_fib, fib, luc, zeros_t, N, weights, is_fib=True)
y_real_luc, y_imag_luc = output_attention(y_real_luc, y_imag_luc, fib, luc, zeros_t, N, weights, is_fib=False)
y_real = y_real_fib + y_real_luc
y_imag = y_imag_fib + y_imag_luc
print(f"Toroidal Nodule Output: {phasor_to_token(y_real, y_imag, vocab)}")
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