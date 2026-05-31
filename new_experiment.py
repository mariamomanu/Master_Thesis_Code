import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import pickle
import sys
import os
from datetime import datetime
from feature_extraction import FeatureExtractor

# hardcoded settings
SAMPLE_RATE = 44100
START_FREQ = 70
END_FREQ = 5000
DURATION = 5
MODEL_FILE = "best_model_water.pkl"
MODEL_FILE_CALIBRATED = "best_model_calibrated.pkl"
CALIBRATION_FILE = "calibration_baseline.npy"
RECORDING_LOCATION = "recordings_experiments"


# sweep
def generate_sweep():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    k = (END_FREQ / START_FREQ) ** (1 / DURATION)
    phase = 2 * np.pi * START_FREQ * (k ** t - 1) / np.log(k)
    return (0.3 * np.sin(phase)).astype(np.float32)


def record():
    sweep = generate_sweep()
    recorded = sd.playrec(sweep, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    recorded = recorded.flatten()
    return recorded


def calibrate():
    print("\n" + "=" * 100)
    print("CALIBRATION — Empty Vessel")
    print("=" * 100)
    print("Empty the vessel completely, then press Enter.")
    input()

    print("Playing sweep")
    recorded = record()
    recorded = recorded / (np.max(np.abs(recorded)) + 1e-10)
    extractor = FeatureExtractor(sample_rate=44100)
    features = {}
    features.update(extractor.extract_time_domain_features(recorded))
    features.update(extractor.extract_spectral_features(recorded))
    features.update(extractor.extract_mfcc_features(recorded, n_mfcc = 13))
    features.update(extractor.extract_frequency_band_energy(recorded))
    features.update(extractor.extract_resonance_peaks(recorded, n_peaks = 5))
    baseline = np.array(list(features.values()))
    np.save(CALIBRATION_FILE, baseline)

    print(f"Baseline RMS: {np.sqrt(np.mean(recorded**2)):.4f}")
    print("Calibration saved.")
    return baseline


def load_or_calibrate():
    if os.path.exists(CALIBRATION_FILE):
        print("\nFound existing calibration.")
        print("r = recalibrate")
        print("Enter = use existing")
        choice = input("Choice: ").strip().lower()
        if choice != 'r':
            baseline = np.load(CALIBRATION_FILE)
            print("Calibration loaded.")
            return baseline

    return calibrate()



def main():
    print("\n" + "=" * 100)
    print("Liquid Level Measurement")
    print("=" * 100)

    # inputs
    material = input("Vessel material (e.g. glass, plastic): ").strip()

    print("\nMeasurement type:")
    print("1. Liquid level")
    print("2. Liquid-liquid interface")
    mtype = input("Type (1-2): ").strip()

    if mtype == "2":
        top = input("Top liquid (e.g. oil): ").strip()
        bot = input("Bottom liquid (e.g. water): ").strip()
        liquid = f"{top}/{bot}"
    else:
        liquid = input("Liquid type (e.g. water, oil): ").strip()

    # calibration mode (which model to load)
    print("\nCalibration mode:")
    print("1. With calibration (more accurate, requires empty vessel measurement)")
    print("2. Without calibration (quicker, no empty vessel needed)")
    mode = input("Choice (1-2): ").strip()

    if mode == "1":
        model_file = MODEL_FILE_CALIBRATED
        baseline = load_or_calibrate()
    else:
        model_file = MODEL_FILE
        baseline = None

    # load model
    if not os.path.exists(model_file):
        print(f"\nERROR: {model_file} not found.")
        sys.exit(1)

    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    model  = model_data['model']
    scaler = model_data['scaler']

    # measure
    print("\nAdd your liquid, then press Enter when ready.")
    input()
    print(f"Playing sweep ({START_FREQ}-{END_FREQ} Hz, {DURATION}s)")
    recorded = record()
    print("Recording done.")

    # extract -> subtract baseline -> scale -> predict
    print("Extracting features")
    recorded = recorded / (np.max(np.abs(recorded)) + 1e-10)
    extractor = FeatureExtractor(sample_rate = 44100)
    features = {}
    features.update(extractor.extract_time_domain_features(recorded))
    features.update(extractor.extract_spectral_features(recorded))
    features.update(extractor.extract_mfcc_features(recorded, n_mfcc = 13))
    features.update(extractor.extract_frequency_band_energy(recorded))
    features.update(extractor.extract_resonance_peaks(recorded, n_peaks = 5))
    feat_vec = np.array(list(features.values()))
    if baseline is not None:
        feat_vec = feat_vec - baseline
    feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
    pred = float(np.clip(model.predict(feat_scaled)[0], 0, 100))

    # save recording
    os.makedirs(RECORDING_LOCATION, exist_ok = True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{RECORDING_LOCATION}/recording_{ts}.wav"
    wav.write(fname, SAMPLE_RATE, recorded)

    # results
    print("\n" + "=" * 100)
    print(f"Vessel material : {material}")
    print(f"Liquid : {liquid}")
    print(f"Type : {'Interface' if mtype == '2' else 'Liquid level'}")
    print("-" * 100)
    print(f"PREDICTED LEVEL : {pred:.1f}%")
    print(f"Output RMS : {np.sqrt(np.mean(recorded**2)):.4f}")
    print(f"Recording saved : {fname}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()