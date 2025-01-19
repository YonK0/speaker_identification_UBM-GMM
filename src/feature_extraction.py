import numpy as np
import librosa
from scipy.fftpack import dct
from scipy.signal import lfilter
from scipy.signal.windows import hamming

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, frame_size=0.025, frame_stride=0.01, 
                 preemphasis_coef=0.97, num_filters=40, num_ceps=13,
                 min_freq=0, max_freq=None):
        """
        Initialize the feature extractor with configurable parameters
        
        Args:
            sample_rate (int): Sample rate of the audio signal
            frame_size (float): Size of each frame in seconds
            frame_stride (float): Stride between frames in seconds
            preemphasis_coef (float): Preemphasis coefficient
            num_filters (int): Number of mel filters
            num_ceps (int): Number of cepstral coefficients
            min_freq (int): Minimum frequency for mel filters
            max_freq (int): Maximum frequency for mel filters
        """
        self.sample_rate = sample_rate
        self.frame_size = int(frame_size * sample_rate)
        self.frame_stride = int(frame_stride * sample_rate)
        self.preemphasis_coef = preemphasis_coef
        self.num_filters = num_filters
        self.num_ceps = num_ceps
        self.min_freq = min_freq
        self.max_freq = max_freq if max_freq else sample_rate // 2
        
    def load_audio(self, file_path):
        """Load audio file and resample if necessary."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
        
    def preemphasis(self, signal):
        """
        Apply preemphasis filter to the signal.
        y(n) = x(n) - α * x(n-1)
        """
        return np.append(signal[0], signal[1:] - self.preemphasis_coef * signal[:-1])
        
    def framing(self, signal):
        """
        Split signal into frames using sliding window approach.
        """
        frame_length = self.frame_size
        frame_step = self.frame_stride
        signal_length = len(signal)
        
        # Calculate number of frames
        num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1
        
        # Pad signal to ensure all frames are full
        pad_length = (num_frames - 1) * frame_step + frame_length
        pad_signal = np.pad(signal, (0, pad_length - signal_length))
        
        # Create indices for frames
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                 np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
                 
        return pad_signal[indices.astype(np.int32, copy=False)]
        
    def apply_window(self, frames):
        """
        Apply Hamming window to frames.
        """
        return frames * hamming(self.frame_size)
        
    def power_spectrum(self, frames):
        """
        Calculate power spectrum using FFT.
        """
        # Compute FFT
        mag_frames = np.absolute(np.fft.rfft(frames, n=self.frame_size))
        
        # Compute power spectrum
        pow_frames = (mag_frames ** 2) / self.frame_size
        
        return pow_frames
        
    def mel_filterbank(self):
        """
        Create mel filterbank.
        """
        # Convert min and max frequencies to mel scale
        low_freq_mel = 2595 * np.log10(1 + self.min_freq / 700)
        high_freq_mel = 2595 * np.log10(1 + self.max_freq / 700)
        
        # Generate mel points
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.num_filters + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Convert to FFT bin numbers
        fft_bin = np.floor((self.frame_size + 1) * hz_points / self.sample_rate)
        
        # Create filterbank
        filterbank = np.zeros([self.num_filters, self.frame_size // 2 + 1])
        
        for m in range(1, self.num_filters + 1):
            f_m_minus = int(fft_bin[m - 1])
            f_m = int(fft_bin[m])
            f_m_plus = int(fft_bin[m + 1])
            
            for k in range(f_m_minus, f_m):
                filterbank[m - 1, k] = (k - fft_bin[m - 1]) / (fft_bin[m] - fft_bin[m - 1])
            for k in range(f_m, f_m_plus):
                filterbank[m - 1, k] = (fft_bin[m + 1] - k) / (fft_bin[m + 1] - fft_bin[m])
                
        return filterbank
        
    def extract_features(self, audio_path):
        """
        Extract features following the complete pipeline.
        """
        # Load audio
        signal = self.load_audio(audio_path)
        
        # Apply pre-emphasis
        emphasized_signal = self.preemphasis(signal)
        
        # Frame the signal
        frames = self.framing(emphasized_signal)
        
        # Apply Hamming window
        windowed_frames = self.apply_window(frames)
        
        # Compute power spectrum
        power_spec = self.power_spectrum(windowed_frames)
        
        # Apply mel filterbank
        filterbank = self.mel_filterbank()
        mel_spec = np.dot(power_spec, filterbank.T)
        
        # Take log
        log_mel_spec = np.log(mel_spec + 1e-8)
        
        # Apply DCT
        mfcc = dct(log_mel_spec, type=2, axis=1, norm='ortho')[:, :self.num_ceps]
        
        # Calculate delta features
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Combine all features
        combined_features = np.hstack([
            mfcc,
            delta1,
            delta2,
            np.mean(log_mel_spec, axis=1, keepdims=True),  # Energy feature
            np.std(log_mel_spec, axis=1, keepdims=True)    # Spectral variance
        ])
        
        return combined_features

    # def get_feature_dimension(self):
    #     """Return the dimension of the feature vector."""
    #     return self.num_ceps * 3 + 2  # MFCC + delta1 + delta2 + energy + variance