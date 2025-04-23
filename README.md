#  Oculus_HGR - Hand Gesture Recognition System

A real-time hand gesture recognition system using **MediaPipe** and **XGBoost**. 
Easily add new gestures, retrain your model, and recognize gestures via webcam.

---

## Project Structure
Oculus_HGR/
├── data/                    # Raw and merged gesture CSV files
├── models/                  # Trained model, scaler, and label encoder
├── scripts/                 # All functional scripts
│   ├── add_samples.py       # Collect new gesture samples via webcam
│   ├── merge_data.py        # Merge all CSVs into one dataset
│   ├── train_model.py       # Train the XGBoost model
│   ├── check_labels.py      # View known gesture labels
│   └── run_realtime.py      # Run real-time gesture recognition
├── main.py                  # Optional menu-based launcher (see below)
├── requirements.txt         # Python dependencies
└── README.md                # Project overview and usage

##  How to Use

### 1. Install dependencies

run the below command in the terminal:

pip install -r requirements.txt

---

### 2. Collect gesture samples

```bash
python scripts/add_samples.py
```

- Press `s` to start collecting.
- Press `q` to stop webcam.

---

### 3. Merge all gesture CSVs into one file

```bash
python scripts/merge_data.py
```

---

### 4. Train the model

```bash
python scripts/train_model.py
```

---

### 5. Check known gesture labels (optional)

```bash
python scripts/check_labels.py
```

---

### 6. Run real-time prediction

```bash
python scripts/run_realtime.py
```

---

## Add New Gestures

To add more gestures later:

1. Run `add_samples.py` with the new gesture label.
2. Run `merge_data.py` again.
3. Retrain the model with `train_model.py`.
4. Done! New gestures are live.

---

##  Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy, Pandas
- scikit-learn
- XGBoost
- joblib

All handled via:

```bash
pip install -r requirements.txt
```

---

that is all for now I plan to add more gesture and increase the dataset
and add a GUI if the need arise