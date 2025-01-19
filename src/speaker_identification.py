import os
import numpy as np
from scipy.io import wavfile
import librosa
import python_speech_features
from sklearn.mixture import GaussianMixture
import joblib
#Custom
from feature_extraction import *

class SpeakerIdentification:
    def __init__(self, n_components=128):
        self.n_components = n_components
        self.feature_extractor = AudioFeatureExtractor()
        self.ubm = None
        self.speaker_models = {}
        
    def extract_features(self, audio_path):
        """Extract features using the AudioFeatureExtractor."""
        return self.feature_extractor.extract_features(audio_path)

    def train_ubm(self, data_dir):
        """Train Universal Background Model using all available data."""
        all_features = []
        
        # Collect features from all speakers
        for speaker in os.listdir(data_dir):
            speaker_dir = os.path.join(data_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
                
            for audio_file in os.listdir(speaker_dir):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(speaker_dir, audio_file)
                    features = self.extract_features(audio_path)
                    all_features.append(features)
        
        # Concatenate all features
        all_features = np.vstack(all_features)
        
        # Train UBM
        print("Training UBM...")
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            n_init=5,
            random_state=42
        )
        self.ubm.fit(all_features)
        print("UBM training completed")
        
    def adapt_model(self, features, ubm):
        """Perform MAP adaptation for speaker-specific model."""
        # Relevance factor for MAP adaptation
        relevance_factor = 16.0
        
        # Get statistics from UBM
        responsibilities = ubm.predict_proba(features)
        
        # Calculate sufficient statistics
        n_k = responsibilities.sum(axis=0)
        f_k = np.dot(responsibilities.T, features)
        
        # Calculate adaptation coefficients
        alpha_k = n_k / (n_k + relevance_factor)
        
        # Adapt means
        adapted_means = (alpha_k[:, np.newaxis] * (f_k / n_k[:, np.newaxis])) + \
                       ((1 - alpha_k[:, np.newaxis]) * ubm.means_)
        
        # Create adapted model
        adapted_model = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            random_state=42
        )
        
        # Set parameters
        adapted_model.means_ = adapted_means
        adapted_model.covariances_ = ubm.covariances_
        adapted_model.weights_ = ubm.weights_
        adapted_model.precisions_cholesky_ = ubm.precisions_cholesky_
        
        return adapted_model
    
    def train(self, data_dir):
        """Train speaker-specific models using MAP adaptation."""
        # First train UBM if not already trained
        if self.ubm is None:
            self.train_ubm(data_dir)
        
        # Train speaker-specific models
        for speaker in os.listdir(data_dir):
            speaker_dir = os.path.join(data_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            
            speaker_features = []
            for audio_file in os.listdir(speaker_dir):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(speaker_dir, audio_file)
                    features = self.extract_features(audio_path)
                    speaker_features.append(features)
            
            # Concatenate all features for this speaker
            speaker_features = np.vstack(speaker_features)
            
            # Adapt UBM to create speaker-specific model
            print(f"Adapting model for speaker {speaker}")
            self.speaker_models[speaker] = self.adapt_model(speaker_features, self.ubm)
        
        print("Training completed for all speakers")
    
    def identify_speaker(self, audio_path):
        """Identify speaker from audio file."""
        # Extract features from test audio
        features = self.extract_features(audio_path)
        
        # Calculate log-likelihood for each speaker
        scores = {}
        for speaker, model in self.speaker_models.items():
            scores[speaker] = np.mean(model.score_samples(features))
        
        # Return speaker with highest score
        identified_speaker = max(scores.items(), key=lambda x: x[1])[0]
        return identified_speaker, scores
    
    def save_models(self, path):
        """Save trained models to disk."""
        models_dict = {
            'ubm': self.ubm,
            'speaker_models': self.speaker_models,
            'n_components': self.n_components,
            'n_mfcc': self.feature_extractor
        }
        joblib.dump(models_dict, path)
    
    @classmethod
    def load_models(cls, path):
        """Load trained models from disk."""
        models_dict = joblib.load(path)
        
        # Create a new instance with loaded parameters
        identifier = cls(n_components=models_dict['n_components'])
        
        # Set the models
        identifier.ubm = models_dict['ubm']
        identifier.speaker_models = models_dict['speaker_models']
        
        return identifier