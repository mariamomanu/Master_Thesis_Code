import numpy as np
import pandas as pd
import librosa
import scipy.io.wavfile as wav
from scipy import signal, stats
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# global variables

VIZ_LOCATION = 'feature_analysis.png'
INPUT_CSV_FILE ='experiment_data.csv'
OUTPUT_CSV_FILE ='extracted_features.csv'


class FeatureExtractor:
    def __init__(self, sample_rate = 44100):
        self.sample_rate = sample_rate

    # extract frequency domain features
    def extract_spectral_features(self, y):
        features = {}
        
        # spectral centroid (center of mass of spectrum)
        centroid = librosa.feature.spectral_centroid(y = y, sr = self.sample_rate)
        features['spectral_centroid_mean'] = np.mean(centroid)
        features['spectral_centroid_std'] = np.std(centroid)
        
        # spectral rolloff (frequency below which 85% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y = y, sr = self.sample_rate, roll_percent= 0.85)
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        features['spectral_rolloff_std'] = np.std(rolloff)
        
        # spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y = y, sr = self.sample_rate)
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        features['spectral_bandwidth_std'] = np.std(bandwidth)
        
        # spectral flatness (measure of noisiness)
        flatness = librosa.feature.spectral_flatness(y = y)
        features['spectral_flatness_mean'] = np.mean(flatness)
        features['spectral_flatness_std'] = np.std(flatness)
        
        # zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)
        
        return features

    # MFCC features (signal shape)
    def extract_mfcc_features(self, y, n_mfcc = 13):
        features = {}
        
        mfccs = librosa.feature.mfcc(y = y, sr = self.sample_rate, n_mfcc = n_mfcc)
        
        # statistics of each MFCC coefficient
        for i in range(n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        return features

    # energy in different frequency segments
    def extract_frequency_band_energy(self, y, bands=None):
        if bands is None:
            # frequency segments in Hz from low to high
            bands = [
                (70, 200),
                (200, 500),
                (500, 1000),
                (1000, 2000),
                (2000, 3500),
                (3500, 5000)
            ]
        
        features = {}
        
        # compute FFT
        fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1/self.sample_rate)
        power = np.abs(fft) ** 2
        
        # energy in each band
        for i, (low, high) in enumerate(bands):
            mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(power[mask])
            features[f'energy_band_{low}_{high}Hz'] = band_energy
        
        # normalize by total energy
        total_energy = np.sum(power)
        for i, (low, high) in enumerate(bands):
            mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(power[mask])
            features[f'energy_ratio_{low}_{high}Hz'] = band_energy / (total_energy + 1e-10)
        
        return features



    # dominant resonance frequencies
    def extract_resonance_peaks(self, y, n_peaks = 5):

        features = {}
        
        # computing power spectral density
        freqs, psd = signal.welch(y, self.sample_rate, nperseg = 2048)
        
        # find peaks
        peaks, properties = signal.find_peaks(psd, height=np.mean(psd), distance = 10)
        
        # sort by height
        if len(peaks) > 0:
            peak_heights = properties['peak_heights']
            sorted_indices = np.argsort(peak_heights)[::-1]
            
            for i in range(n_peaks):
                if i < len(sorted_indices):
                    idx = sorted_indices[i]
                    peak_idx = peaks[idx]
                    features[f'resonance_freq_{i+1}'] = freqs[peak_idx]
                    features[f'resonance_amplitude_{i+1}'] = peak_heights[idx]
                else:
                    features[f'resonance_freq_{i+1}'] = 0
                    features[f'resonance_amplitude_{i+1}'] = 0
        else:
            for i in range(n_peaks):
                features[f'resonance_freq_{i+1}'] = 0
                features[f'resonance_amplitude_{i+1}'] = 0
        
        return features

    # time domain statistical features
    def extract_time_domain_features(self, y):

        features = {}
        
        # statistics
        features['rms_energy'] = np.sqrt(np.mean(y**2))
        features['peak_amplitude'] = np.max(np.abs(y))
        features['mean_amplitude'] = np.mean(np.abs(y))
        features['std_amplitude'] = np.std(y)
        
        # higher-order statistics
        features['skewness'] = stats.skew(y)
        features['kurtosis'] = stats.kurtosis(y)
        
        # dynamic range
        features['dynamic_range'] = 20 * np.log10(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10))
        
        return features

    # extract all features from the audio file
    def extract_all_features(self, audio_file):

        try:
            # load audio
            sr, y = wav.read(audio_file)
            
            # convert to mono if stereo
            if len(y.shape) > 1:
                y = y[:, 0]
            
            # normalize
            y = y.astype(np.float32)
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # extract all feature groups
            features = {}
            features.update(self.extract_time_domain_features(y))
            features.update(self.extract_spectral_features(y))
            features.update(self.extract_mfcc_features(y, n_mfcc = 13))
            features.update(self.extract_frequency_band_energy(y))
            features.update(self.extract_resonance_peaks(y, n_peaks =5))
            
            return features
        
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return None


def process_dataset(csv_file, output_file):
    
    # load metadata
    print(f"\nLoading metadata from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"{len(df)} recordings")
    
    # check data distribution
    print("\nData distribution:")
    print(df['liquid_level'].value_counts().sort_index())

    # initialize f exractor
    extractor = FeatureExtractor(sample_rate = 44100)
    
    print("\nExtracting features")
    all_features = []
    
    for idx, row in tqdm(df.iterrows(), total = len(df)):
        audio_file = row['filename']
        
        if not os.path.exists(audio_file):
            print(f"File not found - {audio_file}")
            continue
        
        # extract features
        features = extractor.extract_all_features(audio_file)
        
        if features is not None:
            # add metadata
            features['liquid_level'] = row['liquid_level']
            features['container_type'] = row['container_type']
            features['liquid_type'] = row['liquid_type']
            features['timestamp'] = row['timestamp']
            features['filename'] = audio_file
            
            all_features.append(features)
    
    # create dataframe
    features_df = pd.DataFrame(all_features)
    
    # reorder columns (put target variable first)
    cols = ['liquid_level'] + [col for col in features_df.columns if col != 'liquid_level']
    features_df = features_df[cols]

    features_df.to_csv(output_file, index = False)

    
    # show feature statistics
    print("\n" + "="*100)
    print("FEATURE SUMMARY")
    print("="*100)
    
    # exclude metadata columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['liquid_level', 'container_type', 'liquid_type', 'timestamp', 'filename']]
    
    print(f"\nTotal features extracted: {len(feature_cols)}")
    print("\nFeature groups:")
    
    feature_groups = {
        'Time domain': [col for col in feature_cols if any(x in col for x in ['rms', 'peak', 'mean', 'std', 'skew', 'kurt', 'dynamic'])],
        'Spectral': [col for col in feature_cols if 'spectral' in col or 'zero_crossing' in col],
        'MFCC': [col for col in feature_cols if 'mfcc' in col],
        'Energy bands': [col for col in feature_cols if 'energy' in col],
        'Resonances': [col for col in feature_cols if 'resonance' in col]
    }
    
    for group_name, group_features in feature_groups.items():
        print(f"  {group_name}: {len(group_features)} features")
    
    # show sample
    print("\n" + "="*100)
    print("SAMPLE DATA (first 5 rows, key features)")
    print("="*100)
    sample_cols = ['liquid_level', 'rms_energy', 'spectral_centroid_mean', 
                   'resonance_freq_1', 'energy_ratio_200_500Hz']
    sample_cols = [col for col in sample_cols if col in features_df.columns]
    print(features_df[sample_cols].head())
    
    return features_df


def analyze_features(features_df):
    
    # exclude metadata
    feature_cols = [col for col in features_df.columns 
                   if col not in ['liquid_level', 'container_type', 'liquid_type', 'timestamp', 'filename']]
    
    # correlation with target, except self-correlation
    print("\nTop 10 features correlated with liquid level:")
    correlations = features_df[feature_cols + ['liquid_level']].corr()['liquid_level'].abs().sort_values(ascending=False)
    print(correlations[1:11])
    
    # plotesome of the key features vs liquid level
    fig, axes = plt.subplots(2, 3, figsize = (15, 10))
    
    key_features = [
        'rms_energy',
        'spectral_centroid_mean',
        'resonance_freq_1',
        'energy_ratio_200_500Hz',
        'mfcc_1_mean',
        'spectral_rolloff_mean'
    ]
    
    for idx, feature in enumerate(key_features):
        if feature in features_df.columns:
            ax = axes[idx // 3, idx % 3]
            
            # make scatter plots
            for level in sorted(features_df['liquid_level'].unique()):
                data = features_df[features_df['liquid_level'] == level][feature]
                ax.scatter([level] * len(data), data, alpha = 0.5, s = 30)
            
            ax.set_xlabel('Liquid Level (%)')
            ax.set_ylabel(feature)
            ax.set_title(f'{feature} vs Liquid Level')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_LOCATION, dpi = 300, bbox_inches = 'tight')
    print(f"\nFeature analysis plot saved.")
    plt.show()


if __name__ == "__main__":
    
    # process dataset and extract features
    features_df = process_dataset(INPUT_CSV_FILE, OUTPUT_CSV_FILE)
    
    # analyze features
    print("\n" + "="*100)
    analyze_choice = input("Analyze and plot features? (y/n): ").lower()
    if analyze_choice == 'y':
        analyze_features(features_df)
