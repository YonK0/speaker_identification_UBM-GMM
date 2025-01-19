import pyaudio
import wave
from pynput import keyboard

def record_voice(output_filename="output.wav", sample_rate=16000, channels=1, chunk_size=4000):
    """
    Records audio from the microphone and saves it to a .wav file.
    Press 'q' to stop recording.

    Args:
        output_filename (str): The name of the output .wav file.
        sample_rate (int): Sampling rate of the recording (e.g., 44100 Hz).
        channels (int): Number of audio channels (1 for mono, 2 for stereo).
        chunk_size (int): Size of each audio chunk.
    """
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open the stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording... Press 'q' to stop.")

    frames = []
    stop_recording = False

    # Define a callback to stop recording on 'q' press
    def on_press(key):
        nonlocal stop_recording
        try:
            if key.char == 'q':
                stop_recording = True
                return False  # Stop listener
        except AttributeError:
            pass

    # Start the listener in a blocking manner
    with keyboard.Listener(on_press=on_press) as listener:
        while not stop_recording:
            data = stream.read(chunk_size)
            frames.append(data)

        listener.join()

    print("Recording stopped.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio instance
    audio.terminate()

    # Save the audio as a .wav file
    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    return output_filename