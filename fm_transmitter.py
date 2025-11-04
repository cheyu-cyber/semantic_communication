# fm_transmitter.py
import numpy as np
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import soundfile as sf
import time
from scipy.signal import resample

# ========= PARAMETERS =========
CENTER_FREQ = 99.5e6       # Tune this to your desired frequency
TX_GAIN = 10
CHUNK_SIZE = 1024
FM_DEVIATION = 75e3        # 75 kHz deviation for wideband FM

# ========= READ AUDIO =========
print("Reading Audio")
audio, fs = sf.read("test.wav")
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # make mono

# Resample to match SDR sample rate
print(f"Sampling rate is {fs}")
SAMPLE_RATE = (2_000_000 // fs)*fs    # 2 MHz
print("Resampling")
num_samples = int(len(audio) * SAMPLE_RATE / fs)
audio_resampled = resample(audio, num_samples)

# ========= FM MODULATION =========
# Integrate audio to phase, then make complex baseband
print("FM Modulation")
kf = 2 * np.pi * FM_DEVIATION / SAMPLE_RATE
phase = np.cumsum(kf * audio_resampled)
iq = np.exp(1j * phase).astype(np.complex64)

# ========= SDR SETUP =========
print("Device Setup")
sdr = SoapySDR.Device(dict(driver="hackrf"))
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)

tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
sdr.activateStream(tx_stream)

print(f"Transmitting {len(iq)/SAMPLE_RATE:.1f} seconds of audio at {CENTER_FREQ/1e6:.1f} MHz...")

# ========= STREAM =========
for i in range(0, len(iq), CHUNK_SIZE):
    chunk = iq[i:i+CHUNK_SIZE]
    if len(chunk) < CHUNK_SIZE:
        chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
    sdr.writeStream(tx_stream, [chunk], len(chunk))

sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)
print("Done transmitting.")

