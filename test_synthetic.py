from src.synthetic_data import generate_synthetic_patient_data

# Call the function
result = generate_synthetic_patient_data(10)

print("Type:", type(result))
print("Shape:", result.shape if hasattr(result, "shape") else "No shape")
print("First 5 rows:\n", result[:5])
