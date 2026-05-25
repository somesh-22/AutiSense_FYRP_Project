from phase1_screening import train_phase1_model

results = train_phase1_model()

print("\nMODEL TRAINED SUCCESSFULLY\n")

print("Accuracy:", results["accuracy"])

print("\nClassification Report:\n")
print(results["report"])

print("\nConfusion Matrix:\n")
print(results["confusion_matrix"])