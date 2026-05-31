import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile as wav
from datetime import datetime
import pandas as pd
import os


# global variables - change accordingly

NUM_EXPERIMENTS = 10
CSV_FILE = "experiment_data.csv"
RECORDING_LOCATION = "recordings"
# visualize - recommend setting it to True it in for sanity checks at different levels before actually starting data collection, 
# with NUM_EXPERIMENTS set at a low number
DO_VISUALIZE = True

# metadata - change accordingly

# liquid level (level or interface; if interface, the level means the level of the first liquid mentioned, with the other liquid 
# contributing to 100% fill)
LIQUID_LEVEL = "100"
# container type (glass/plastic/metal)
CONTAINER_TYPE = "glass"
# liquid type (water/oil/etc)
LIQUID_TYPE = "water"

# frequency sweep data - if changed, all models need to be retrained with new data at the new freq. sweep

SAMPLE_RATE = 44100
START_FREQ = 70
END_FREQ = 5000
DURATION = 5

os.makedirs(RECORDING_LOCATION, exist_ok=True)
# generate frequency sweep, inspired by Luca Lu, in https://medium.com/@b04502057/sweep-chirp-generator-07edb0e951fa
def generate_sweep(start_freq, end_freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration))
    k = (end_freq / start_freq) ** (1 / duration)
    phase = 2 * np.pi * start_freq * (k ** t - 1) / np.log(k)
    # amplitude 0.3
    sweep = 0.3 * np.sin(phase) 
    return sweep.astype(np.float32), t


# run one test, show the sweep and record
def run_test():
    print("RUNNING TEST")
    
    # generate the frequency sweep
    print(f"\nGenerating sweep: {START_FREQ}-{END_FREQ} Hz, {DURATION}s")
    input_signal, time_array = generate_sweep(START_FREQ, END_FREQ, DURATION, SAMPLE_RATE)
    
    # play and record
    print("Playing and recording starting.")
    output_signal = sd.playrec(input_signal, samplerate = SAMPLE_RATE, channels = 1, dtype = 'float32')
    sd.wait()
    print("Playing and recording finished.")
    
    output_signal = output_signal.flatten()
    
    # save audio for future feature extraction
    date = datetime.now()
    timestamp = date.strftime("%Y%m%d_%H%M%S")
    audio_filename = f"{RECORDING_LOCATION}/recording_{timestamp}.wav"
    wav.write(audio_filename, SAMPLE_RATE, output_signal)
    
    # save metadata
    metadata = {
        'timestamp': date.isoformat(),
        'liquid_level': LIQUID_LEVEL,
        'container_type': CONTAINER_TYPE,
        'liquid_type': LIQUID_TYPE,
        'start_freq': START_FREQ,
        'end_freq': END_FREQ,
        'duration': DURATION,
        'filename': audio_filename
    }
    
    # add data to csv
    df = pd.DataFrame([metadata])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode = 'a', header = False, index = False)
    else:
        df.to_csv(CSV_FILE, index = False)
    
    print(f"\nSaved: {audio_filename}")
    print(f"Metadata saved to: {CSV_FILE}")

    if DO_VISUALIZE is True:
        visualize(input_signal, output_signal, time_array)
    
    return metadata

# see input and output side by side
def visualize(input_signal, output_signal, time_array):
    fig, axes = plt.subplots(3, 1, figsize=(20, 14))

    # waveforms
    axes[0].plot(time_array, input_signal, 'b-', alpha = 0.6, label='Input', linewidth = 1.5)
    axes[0].plot(time_array, output_signal, 'r-', alpha = 0.6, label='Output', linewidth= 1.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Waveforms - Input vs Output', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # frequency spectra
    freqs_in,  psd_in  = signal.welch(input_signal,  SAMPLE_RATE, nperseg = 2048)
    freqs_out, psd_out = signal.welch(output_signal, SAMPLE_RATE, nperseg = 2048)
    axes[1].semilogy(freqs_in,  psd_in,  'b-', alpha = 0.6, label = 'Input',  linewidth = 1.5)
    axes[1].semilogy(freqs_out, psd_out, 'r-', alpha = 0.6, label = 'Output', linewidth = 1.5)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density')
    axes[1].set_title('Frequency Spectrum - Input vs Output', fontweight='bold')
    axes[1].set_xlim([0, END_FREQ * 1.2])
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    # spectrogram
    f, t, Sxx = signal.spectrogram(output_signal, SAMPLE_RATE, nperseg = 1024)
    im = axes[2].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading= 'gouraud', cmap= 'viridis')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Spectrogram - Output Signal', fontweight = 'bold')
    axes[2].set_ylim([0, END_FREQ * 1.2])
    plt.colorbar(im, ax= axes[2], label = 'Power (dB)')

    plt.tight_layout(pad=3.0)
    plt.show()

def main():
    print("\n" + "="*100)
    print("VIBRATION RECORDING")
    print("="*100)
    print(f"\nSettings:")
    print(f"Frequency range: {START_FREQ}-{END_FREQ} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    
    # show audio devices
    print("\n" + "="*100)
    print("AVAILABLE AUDIO DEVICES")
    print("="*100)
    print(sd.query_devices())
    
    # run tests 
    for i in range(NUM_EXPERIMENTS):
        print(f"Datapoint {i} at fill level: {LIQUID_LEVEL} for liquid / liquid combo: {LIQUID_TYPE} in vessel: {CONTAINER_TYPE}")
        run_test()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStop")
    except Exception as e:
        print(f"\nError: {e}")
