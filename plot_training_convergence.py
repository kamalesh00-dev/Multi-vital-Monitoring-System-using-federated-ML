import matplotlib.pyplot as plt

# Manually extracted from training log (global val accuracy)
rounds = list(range(1, 51))
global_val_acc = [
    0.9475, 0.9483, 0.9497, 0.9514, 0.9538,
    0.9516, 0.9527, 0.9541, 0.9536, 0.9534,
    0.9547, 0.9542, 0.9533, 0.9551, 0.9560,
    0.9545, 0.9550, 0.9533, 0.9556, 0.9577,
    0.9562, 0.9545, 0.9558, 0.9611, 0.9549,
    0.9581, 0.9551, 0.9566, 0.9569, 0.9546,
    0.9556, 0.9555, 0.9556, 0.9605, 0.9557,
    0.9640, 0.9554, 0.9576, 0.9557, 0.9559,
    0.9518, 0.9553, 0.9531, 0.9619, 0.9560,
    0.9572, 0.9543, 0.9572, 0.9570, 0.9616
]

plt.figure(figsize=(7,4))
plt.plot(rounds, global_val_acc, marker='o')
plt.xlabel("Federated Training Rounds")
plt.ylabel("Global Validation Accuracy")
plt.title("Federated Learning Convergence Across Rounds")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/figures/federated_convergence.png", dpi=300)
plt.show()
