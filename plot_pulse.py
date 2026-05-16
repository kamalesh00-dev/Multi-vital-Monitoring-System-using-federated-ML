import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/health_status/health.csv")

plt.figure(figsize=(8,5))
for pid, group in df.groupby("patient_id"):
    plt.plot(group.index, group["pulse"], label=f"Patient {pid}")

plt.xlabel("Record Index")
plt.ylabel("Pulse Rate")
plt.title("Pulse Variation Across Patients")
plt.legend()
plt.grid(True)
plt.savefig("pulse_variation.png")
plt.show()
