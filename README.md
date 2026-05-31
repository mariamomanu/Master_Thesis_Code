# Vibration-Based Liquid Level Detection System
 
A non-invasive liquid level monitoring system using mechanical vibration analysis and machine learning. A logarithmic frequency sweep is played through a tactile transducer attached to the vessel wall, and the vibration response is captured by piezoelectric contact microphones. Machine learning models trained on extracted vibration features predict the liquid level or liquid-liquid interface position without any contact with the liquid.
 
Developed as part of a Master's thesis at the IT University of Copenhagen (2026).
 
## Repository Structure
 
```
├── fetch_training_data.py  # Data acquisition: generates sweep, records response
├── feature_extraction.py   # Extracts 65 features from WAV recordings
├── ml_pipeline.py          # Trains and evaluates RF, SVM, Neural Network
├── new_experiment.py       # Inference app: measure liquid level in real time
├── requirements.txt
└── README.md
```
 
---
 
## Setup
 
### Requirements
 
Python 3.9+
 
```bash
pip install -r requirements.txt
```
 
### Dependencies
 
```
numpy
pandas
scipy
librosa
sounddevice
matplotlib
scikit-learn
tqdm
```
 
### Hardware
 
- Tactile transducer (e.g. Dayton Audio DAEX25CT-4)
- Audio amplifier
- 2× piezoelectric contact microphones (e.g. Schaller Oyster Passive Piezo Pickup)
- Audio interface with simultaneous playback and recording
---
 
## Usage
 
### Step 1 - Collect Training Data
 
Set the experiment metadata at the top of `experiment.py` (liquid level, container type, liquid type) then run:
 
```bash
python experiment.py
```
 
Outputs WAV recordings and a metadata CSV.
 
### Step 2 - Extract Features
 
```bash
python feature_extraction.py
```
 
Processes all recordings listed in the metadata CSV and outputs a feature CSV with 65 numerical features per recording.
 
### Step 3 - Train Models
 
```bash
python ml_pipeline.py
```
 
Trains Random Forest, SVM, and Neural Network regressors with 5-fold stratified cross-validation. Outputs the best model as a `.pkl` file and a performance plot.
 
### Step 4 - Measure a Liquid Level
 
```bash
python new_experiment.py
```
 
Interactive inference app. Prompts for vessel material, measurement type (single liquid or liquid–liquid interface), and calibration mode. Plays the sweep, records the response, and predicts the liquid level.
 
```
====================================================================================================
Liquid Level Measurement
====================================================================================================
Vessel material (e.g. glass, plastic): glass
Measurement type:
1. Liquid level
2. Liquid-liquid interface
Type (1-2): 1
Liquid type (e.g. water, oil): water
 
Calibration mode:
1. With calibration (more accurate, requires empty vessel measurement)
2. Without calibration (quicker, no empty vessel needed)
Choice (1-2): 1
 
PREDICTED LEVEL : 49.8%
Output RMS      : 0.1531
```
 
---
 
## Calibration
 
Calibration records an empty-vessel baseline and subtracts it from all subsequent measurements. This corrects for vessel-specific acoustic offsets and session-to-session variability without retraining.
 
- The baseline is saved and reused across sessions
- Recalibrate when switching vessels or moving the sensors
- Two separate models are needed: one trained with calibration, one without
---
 
## Excitation Signal
 
The logarithmic frequency sweep is defined as:
 
$$s(t) = A \sin\left( \frac{2\pi f_0 (k^t - 1)}{\ln k} \right), \quad k = \left(\frac{f_1}{f_0}\right)^{1/T}$$
 
Default parameters: $f_0 = 70$\,Hz, $f_1 = 5000$\,Hz, $T = 5$\,s, $A = 0.3$.
 
> If the sweep parameters are changed, all models must be retrained on new data collected with the updated sweep.
 
