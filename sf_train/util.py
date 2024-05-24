import librosa
from scipy.io import wavfile
import numpy as np

def read_mono(path, sr = 48000):
    data_sr, data = wavfile.read(path)

    if data.ndim != 1:
        raise ValueError('Only mono audio is supported')

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.float32:
        pass
    elif data.dtype == np.float64:
        data = data.astype(np.float32)
        pass
    else:
        raise ValueError(f'Unsupported data type: {data.dtype}')

    if data_sr != sr:
        print(f'Resampling from {data_sr} to {sr}')
        data = librosa.resample(data, orig_sr = data_sr, target_sr = sr)

    return data

def write_mono(path, samples, sr = 48000):
    wavfile.write(path, sr, samples)
