import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import scipy

# Load the WAV file
def load_wav(filename):
    data, samplerate = sf.read(filename)
    if data.ndim > 1:  # Handle stereo by taking only one channel
        data = data[:, 0]
    return data, samplerate

# Save the WAV file
def save_wav(filename, data, samplerate):
    sf.write(filename, data, samplerate)

# Generate a SSB signal 
def ssb_modulate(signal, carrier_freq, samplerate):
    t = np.arange(len(signal)) / samplerate
    ## cosine part 
    signa_mul_cos = signal * np.cos(2 * np.pi * carrier_freq * t)
    ## hilbert transform
    signal_filter_hilbert = scipy.signal.hilbert(signal)
    siganal_mul_sin = signal_filter_hilbert * np.sin(2 * np.pi * carrier_freq * t)
    ssb_signal = signa_mul_cos - siganal_mul_sin ## SSB signal - upper sideband
    
    return ssb_signal

# Demodulate the SSB signal
def ssb_demodulate(ssb_signal, carrier_freq, samplerate):
    t = np.arange(len(ssb_signal)) / samplerate
    ## cosine part 
    signa_mul_cos = ssb_signal * np.cos(2 * np.pi * carrier_freq * t)
    ## LPF - low pass filter
    filter =np.sin(2 * np.pi * carrier_freq * t)/(np.pi * t)
    filter[t==0] = 2 * carrier_freq
    signal_filter = np.convolve(signa_mul_cos, filter, mode='same')
    return signal_filter

# Real-time recording and processing callback
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status, flush=True)
    
    ssb_signal = ssb_modulate(indata[:, 0], carrier_freq, samplerate)
    recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)
    
    outdata[:, 0] = recovered_signal
    outdata[:, 1] = recovered_signal

# Main function
def main(mode='file', filename=None, duration=5):
    global samplerate
    if mode == 'file' and filename:
        # Load the input WAV file
        data, samplerate = load_wav(filename)
        carrier_freq = samplerate/5

        # Modulate the signal using SSB
        ssb_signal = ssb_modulate(data, carrier_freq, samplerate)

        # Demodulate the signal to recover the original data
        recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)

        # Save the recovered signal to an output WAV file
        output_filename = 'output_' + filename
        save_wav(output_filename, recovered_signal, samplerate)

        # Plotting
        t = np.arange(len(data)) / samplerate

        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(data), 1/samplerate)), np.fft.fftshift(np.abs(np.fft.fft(data))))
        plt.title('Original Baseband Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3,1,2)
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(ssb_signal), 1/samplerate)), np.fft.fftshift(np.abs(np.fft.fft(ssb_signal))))
        plt.title('SSB Modulated Signal (Upper Sideband)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3,1,3)
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(recovered_signal), 1/samplerate)), np.fft.fftshift(np.abs(np.fft.fft(recovered_signal))))
        plt.title('Demodulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()
        plt.tight_layout()

    elif mode == 'live':
        # Start the stream for real-time recording and processing
        with sd.Stream(samplerate=samplerate, channels=2, callback=audio_callback):
            print("Recording and processing in real-time. Press Ctrl+C to stop.")
            sd.sleep(int(duration * 1000))

        # Record a bit of data for plotting purposes
        print("Recording for plotting purposes...")
        recorded_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()

        t = np.arange(len(recorded_data)) / samplerate
        save_wav('recording.wav',recorded_data,samplerate)
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(data), 1/samplerate)), np.fft.fftshift(np.abs(np.fft.fft(data))))
        plt.title('Original Baseband Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3,1,2)
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(ssb_signal), 1/samplerate)), np.fft.fftshift(np.abs(np.fft.fft(ssb_signal))))
        plt.title('SSB Modulated Signal (Upper Sideband)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3,1,3)
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(recovered_signal), 1/samplerate)), np.fft.fftshift(np.abs(np.fft.fft(recovered_signal))))
        plt.title('Demodulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()
        plt.tight_layout()
    else:   
        print("Invalid mode or filename not provided for file mode.")

main(mode='file',filename='file_name.wav')
main(mode='live', duration=5)
