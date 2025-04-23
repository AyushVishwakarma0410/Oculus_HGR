from joblib import load

# Load the saved label encoder
label_encoder = load("models/label_encoder.joblib")

# Print all known gesture labels
print("[+] Known gestures:", label_encoder.classes_)
