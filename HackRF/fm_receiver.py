from SoapySDR import Device, SOAPY_SDR_RX, SOAPY_SDR_CF32  # SOAPY_SDR_ constants
import numpy as np
from scipy.signal import decimate
import wave
import time
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd

# ========== SDR CONFIG ==========
SAMPLE_RATE = 2_000_000      # 2 MHz sampling
CENTER_FREQ = 99.5e6         # 99.5 MHz FM station
AUDIO_DECIMATION = 10         # First stage decimation
AUDIO_SAMPLE_RATE = SAMPLE_RATE // AUDIO_DECIMATION // 5  # ~40 kHz final audio
GAIN = 30
BUFFER_SIZE = 4096

#DURATION = 10
OUTPUT_FILE = "output.wav"

# ========== SDR SETUP ==========
sdr = Device(dict(driver="hackrf"))
sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_RX, 0, GAIN)

rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

#print(f"Receiving FM from {CENTER_FREQ/1e6:.1f} MHz ... Press Ctrl+C to stop.")
#print(f"Recording audio to 'output.wav' at {AUDIO_SAMPLE_RATE} Hz")

# ========== WAV FILE SETUP ==========
wave_file = wave.open("output.wav", "wb")
wave_file.setnchannels(1)
wave_file.setsampwidth(2)  # 16-bit samples
wave_file.setframerate(AUDIO_SAMPLE_RATE)

samples = []

try:
    prev = np.zeros(BUFFER_SIZE,np.complex64)
    while True:
        sr = sdr.readStream(rx_stream, [prev],len(prev))
        if sr.ret > 0:
            fm_demod = np.angle(prev[1:]*np.conj(prev[:-1]))
            audio = decimate(fm_demod, AUDIO_DECIMATION, zero_phase=True)
            audio = np.int16(audio / np.max(np.abs(audio))*32767)
            sd.play(audio, samplerate=SAMPLE_RATE // AUDIO_DECIMATION // 5, blocking=False)
            samples.extend(audio)
except KeyboardInterrupt:
    print("\nStopping receiver...")
finally:
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)


# ================ Capture Loop ================
# samples = []
# start_time = time.time()

# while time.time() - start_time < DURATION:
#     sr = sdr.readStream(rx_stream, [buff], len(buff))
#     if sr.ret > 0:
#         iq = buff[:sr.ret].copy()
        
#         fm_demod = np.angle(iq[1:] * np.conj(iq[:-1]))

#         audio = decimate(fm_demod, AUDIO_DECIMATION, zero_phase=True)
#         samples.extend(audio)

# sdr.deactivateStream(rx_stream)
# sdr.closeStream(rx_stream)

audio = np.array(samples)
audio = audio/np.max(np.abs(audio))
sf.write(OUTPUT_FILE, audio, AUDIO_SAMPLE_RATE)

data,rate = sf.read("output.wav")
plt.figure()
plt.plot(data*1000)
plt.title("Recorded Audio Waveform")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("output_waveform.png")





# # ========== DEMODULATION LOOP ==========
# try:
#     last_phase = 0.0
#     while True:
#         sr = sdr.readStream(rx_stream, [buff], len(buff))
#         if sr.ret > 0:
#             iq = buff[:sr.ret]
#             if (sum(buff != 0)>0):
#                 print("Non-Zero Data")

#             # FM demodulation (phase difference)
#             phase = np.angle(iq)
#             fm_demod = np.diff(np.unwrap(phase))

#             # Decimate to lower sample rate
#             audio = decimate(fm_demod, AUDIO_DECIMATION, zero_phase=True)

#             # Scale and convert to int16 for WAV
#             audio = np.int16(audio / np.max(np.abs(audio)) * 32767)

#             # Write to file
#             wave_file.writeframes(audio.tobytes())

# except KeyboardInterrupt:
#     print("\nStopping and closing file...")

# finally:
#     sdr.deactivateStream(rx_stream)
#     sdr.closeStream(rx_stream)
#     wave_file.close()
#     data,rate = sf.read("output.wav")
#     plt.figure()
#     plt.plot(data*1000)
#     plt.title("Recorded Audio Waveform")
#     plt.xlabel("Sample")
#     plt.ylabel("Amplitude")
#     plt.tight_layout()
#     plt.savefig("output_waveform.png")

#     print("File saved as output.wav")

