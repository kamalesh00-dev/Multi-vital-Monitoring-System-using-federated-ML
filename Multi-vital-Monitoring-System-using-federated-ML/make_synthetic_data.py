import os
import numpy as np
import soundfile as sf

# ======================
# CONFIG
# ======================
SR = 16000          # sample rate
DURATION = 2.0      # seconds
TRAIN_SAMPLES = 100
VALID_SAMPLES = 20
TEST_SAMPLES = 20
SNR_DB = 5.0        # noise level

BASE_DIR = "dataset"

# ======================
# HELPERS
# ======================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def generate_clean_signal(duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Speech-like signal: sum of harmonics
    freqs = np.random.choice([120, 180, 240, 300], size=3, replace=False)
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)

    # Amplitude modulation (speech-like)
    mod = 0.5 * (1 + np.sin(2 * np.pi * np.random.uniform(1, 3) * t))
    signal *= mod

    # Normalize
    signal /= np.max(np.abs(signal)) + 1e-8
    return signal.astype(np.float32)

def add_noise(clean, snr_db):
    noise = np.random.randn(len(clean)).astype(np.float32)
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    scale = np.sqrt(clean_power / (10 ** (snr_db / 10) * noise_power))
    noisy = clean + scale * noise
    noisy /= np.max(np.abs(noisy)) + 1e-8
    return noisy

def generate_split(split, num_samples):
    clean_dir = os.path.join(BASE_DIR, split, "clean")
    noisy_dir = os.path.join(BASE_DIR, split, "noisy")

    ensure_dir(clean_dir)
    ensure_dir(noisy_dir)

    for i in range(num_samples):
        clean = generate_clean_signal(DURATION, SR)
        noisy = add_noise(clean, SNR_DB)

        filename = f"sample_{i:04d}.wav"
        sf.write(os.path.join(clean_dir, filename), clean, SR)
        sf.write(os.path.join(noisy_dir, filename), noisy, SR)

    print(f"✅ Generated {num_samples} samples for '{split}'")

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    generate_split("train", TRAIN_SAMPLES)
    generate_split("valid", VALID_SAMPLES)
    generate_split("test", TEST_SAMPLES)

    print("\n🎉 Synthetic dataset generation complete!")
