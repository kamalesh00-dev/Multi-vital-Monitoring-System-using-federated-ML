import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = np.array([
    [0, 245, 0],
    [0, 178, 0],
    [0,  28, 0]
])

labels = ["Normal", "Moderate", "Critical"]

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Cross-Domain Confusion Matrix (Arrhythmia Dataset)")
plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrix_arrhythmia.png", dpi=300)
plt.show()
