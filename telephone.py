import sys
import numpy as np
from scipy import signal
from pathlib import Path
from typing import Tuple, Optional
from logging import Logger
import soundfile
import librosa


class TelephoneAudioProcess:
    def __init__(
        self, 
        low_freq=300, 
        high_freq=3400, 
        target_sr=8000, 
        window_size=512,
        hop_size=128):
        """
        Initialize the phone filter with frequency parameters.
        
        Args:
            low_freq: Lower frequency bound in Hz
            high_freq: Upper frequency bound in Hz
            target_sr: Target sampling rate in Hz
        """
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.target_sr = target_sr
        self.window_size = window_size
        self.hop_size = hop_size
        self.filename = None
        self.audio_data = None
        self.sample_rate = None
        self.telephone_data = None

    def load_audio(self, filepath: Path):
        """
        Load audio file from disk.
        
        Args:
            filepath: Path to audio file
        """
        waveform, sample_rate = librosa.load(filepath, sr=None)
        self.audio_data = waveform
        self.sample_rate = sample_rate
        self.filename = filepath.name
        print(f"Loaded audio file {self.filename} with samples: {len(self.audio_data)} at {self.sample_rate/1000} kHz")

    def frequency_mask(self, frequencies: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(frequencies)
        mask[(frequencies >= self.low_freq) & (frequencies <= self.high_freq)] = 1

        return mask
    
    def process_audio_array(
            self, 
            audio_data: Optional[np.ndarray] = None, 
            sample_rate: Optional[int] = None
        ) -> Tuple[np.ndarray, int]:
        """
        Process the input raw audio data to apply a old phone filter.
        """
        if audio_data is None:
            audio_data = self.audio_data
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        if self.sample_rate != self.target_sr:
            resampled = librosa.core.resample(
                y=audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.target_sr)
            print(f"Resampled the audio file {self.filename} to {self.target_sr/1000} kHz.")
            self.telephone_data = resampled
        else:
            print(f"The audio file is already at the target sampling rate at {self.target_sr}!")

        spectogram = librosa.core.stft(
            y=self.telephone_data,
            n_fft=self.window_size,
            hop_length=self.hop_size
        )
        print(f"Spectogram of the audio file obtained.")

        frequencies = librosa.fft_frequencies(
            sr=self.target_sr,
            n_fft=self.window_size
        )

        mask = self.frequency_mask(frequencies)
        filtered_spectogram = spectogram * mask[:, np.newaxis]
        
        print(f"Filtered the spectogram of the audio within the masked range.")

        processed = librosa.core.istft(
            filtered_spectogram,
            hop_length=self.hop_size,
            win_length=self.window_size
        )
        self.telephone_data = processed
        print(f"Processed the audio file back into array format.")
    
        return processed, self.target_sr

    def save_audio(self, output_filename: str = None):
        """
        Save processed audio to file.
        
        Args:
            output_filepath: Path where to save the audio
        """
        current_dir = Path.cwd()
        if output_filename is None:
            output_filepath = current_dir / "outputs" / self.filename
        else:
            output_filepath = current_dir / "outputs" / output_filename
            
        soundfile.write(output_filepath, self.telephone_data, self.target_sr)
        print(f"Saved to {output_filepath}")

    def process_file(self, input: str, output: str):
        print("Initiating the telephone audio processor...")
        input_path = Path(input)

        self.load_audio(input_path)
        self.process_audio_array(self.audio_data, self.sample_rate)
        self.save_audio(output)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: telephone.py works as: python telephone.py infile.wav outfile.wav")
    else:
        infile = sys.argv[1]
        outfile = sys.argv[2]
        telephone_processer = TelephoneAudioProcess()

        telephone_processer.process_file(infile, outfile)