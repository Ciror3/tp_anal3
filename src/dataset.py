import glob
import soundfile as sf
from torch.utils.data import Dataset
import torch
import numpy as np
import librosa as lb

class AudioMNISTDataset(Dataset):
    def __init__(self, data_path, feature, test=False):
        self.data_path = data_path
        self.feature = feature
        self.test = test

    def __len__(self):
        if not self.test:
            return len(glob.glob(self.data_path+'/train/*'))
        else:
            return len(glob.glob(self.data_path+'/test/*'))

    def __getitem__(self, idx):
        # Get audio paths
        if not self.test:
            audio_paths = glob.glob(self.data_path + '/train/*')
        else:
            audio_paths = glob.glob(self.data_path + '/test/*')
        
        # Get audio data and labels
        audio, fs = sf.read(audio_paths[idx])
        label = audio_paths[idx].split('/')[-1].split('_')[0].split('\\')[1]
        # Extract features
        if self.feature == 'raw_waveform':
            # insertar código acá
            feat = torch.from_numpy(audio)
        elif self.feature == 'audio_spectrum':
            feat = self.dft(audio,fs)
        elif self.feature == 'mfcc':
            feat = self.mfcc(audio, fs)
        
        #Agrego
        feat =  feat.view(-1)
        if feat.size(0) == 4000:
            feat = torch.cat((feat, torch.ones(1)))
        feat = feat.type(torch.float)
        
        feat = feat.type(torch.float)
        label = torch.tensor(int(label), dtype=torch.long)

        return feat, label

    @staticmethod
    def dft(audio: np.ndarray, fs: float) -> torch.Tensor:
        """
        Calculates the discrete Fourier transform of the audio data, normalizes the result and trims it, preserving only positive frequencies.
        Args:
            audio (Numpy array): audio file to process.
            fs (float): sampling frequency of the audio file.
        Returns:
            audio_f (Tensor): spectral representation of the audio data.
        """
        audio_f = np.fft.rfft(audio) #Transformada de Fourier devuelve parte no negativa
        audio_f = np.abs(audio_f) #/ np.max(np.abs(audio_f)) #Normalizo
        audio_f = torch.from_numpy(audio_f)
        return audio_f

    @staticmethod
    def mfcc(audio, fs):
        """
        Calculates the Mel Frequency Cepstral Coefficients (MFCCs) of the audio data.
        Args:
            audio (Numpy array): audio file to process.
            fs (float): sampling frequency of the audio file.
            mfcc_params (dictionary): the keys are 'n_fft', the length of the FFTs, and 'window', the type of window to be used in the STFTs (see scipy.signal.get_window)
        Returns:
            mfcc (Tensor): MFCC of the input audio file.
        """
        mfcc = lb.feature.mfcc(y=audio, sr=fs, n_mfcc=20)
        mfcc = torch.tensor(mfcc, dtype=torch.float)

        return mfcc
