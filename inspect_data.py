import matplotlib.pyplot as plt
from src.synthetic import generate_synthetic_patient_data


# Generate data
X, y = generate_synthetic_patient_data(100)

# Plot each signal
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(X[:, 0], label="SpO₂")
plt.title("SpO₂ Levels")
plt.xlabel("Sample")
plt.ylabel("Oxygen (%)")
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(X[:, 1], label="ECG", color="orange")
plt.title("ECG Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(X[:, 2], label="Temperature", color="green")
plt.title("Temperature")
plt.xlabel("Sample")
plt.ylabel("°C")
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(X[:, 3], label="Pulse", color="red")
plt.title("Pulse")
plt.xlabel("Sample")
plt.ylabel("BPM")
plt.grid(True)
plt.legend()

plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("data_overview.png")

# Show the plots
plt.show()
