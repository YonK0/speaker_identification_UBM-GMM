import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pyaudio
import wave
import queue
from PIL import Image, ImageTk
from recorder import * 

# Custom
from speaker_identification import *

class SpeakerIdentificationGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Recognition Game                    /esprit/Sagemcom")
        self.root.geometry("400x600")
        
        # Initialize recording variables
        self.recording_thread = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio = None
        
        # Create main container
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        #images
        self.unknown_image = self.load_and_resize_image(f"../images/unknown.jpeg", (200, 200))

        # Title
        self.title_label = ttk.Label(
            self.main_frame,
            text="Who's Speaking?",
            font=('Helvetica', 24, 'bold')
        )
        self.title_label.grid(row=0, column=0, pady=20)
        
        # # Instructions
        # self.instructions = ttk.Label(
        #     self.main_frame,
        #     text="Press 'Start Recording' and speak for a few seconds.\nPress 'Stop Recording' when you're done!",
        #     wraplength=300
        # )
        # self.instructions.grid(row=1, column=0, pady=20)
        
        # Add image label after instructions
        self.image_label = ttk.Label(
            self.main_frame,
            image=self.unknown_image
        )
        self.image_label.grid(row=2, column=0, pady=10)
        
        # Buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=3, column=0, pady=20)
        
        # Start button
        self.record_button = ttk.Button(
            self.button_frame,
            text="Start Recording",
            command=self.start_recording
        )
        self.record_button.grid(row=0, column=0, padx=5)
        
        # Stop button
        self.stop_button = ttk.Button(
            self.button_frame,
            text="Stop Recording",
            command=self.stop_recording,
            state='disabled'
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Status label
        self.status_label = ttk.Label(
            self.main_frame,
            text="Ready to play!",
            font=('Helvetica', 10, 'italic')
        )
        self.status_label.grid(row=4, column=0, pady=10)
        
        # Result frame
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.grid(row=5, column=0, pady=20)
        
        self.result_label = ttk.Label(
            self.result_frame,
            text="",
            font=('Helvetica', 16)
        )
        self.result_label.grid(row=0, column=0, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.main_frame,
            orient="horizontal",
            length=300,
            mode="indeterminate"
        )
        self.progress.grid(row=6, column=0, pady=20)

        # Bind cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_and_resize_image(self, image_path, size):
        """Load and resize an image, return None if image doesn't exist"""
        try:
            image = Image.open(image_path)
            image = image.resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        except:
            # Create a default image if file doesn't exist
            image = Image.new('RGB', size, color='gray')
            return ImageTk.PhotoImage(image)
    
    def on_closing(self):
        """Clean up resources before closing"""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
        if self.audio:
            self.audio.terminate()
        self.root.destroy()
        
    def record_voice(self, output_filename="output.wav", sample_rate=44100, channels=1, chunk_size=1024):
        """Records audio from the microphone and saves it to a .wav file."""
        self.audio = pyaudio.PyAudio()
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        frames = []
        
        while self.is_recording:
            data = stream.read(chunk_size)
            frames.append(data)
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        
        # Save the audio file
        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        
        self.audio_queue.put(output_filename)
    
    def start_recording(self):
        self.is_recording = True
        self.record_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="Recording... Speak now!")
        self.result_label.config(text="")
        self.progress.start()
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.record_voice)
        self.recording_thread.start()
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.stop_button.config(state='disabled')
            self.status_label.config(text="Processing...")
            
            # Wait for recording thread to finish and process the audio
            def process_recording():
                self.recording_thread.join()
                output_filename = self.audio_queue.get()
                
                try:
                    # Initialize speaker identification system
                    speaker_id = SpeakerIdentification.load_models("speaker_models.pkl")
                    
                    # Identify speaker
                    identified_speaker, scores = speaker_id.identify_speaker(output_filename)
                    
                    tempscore = 0
                    # Check confidencecd
                    for _, score in scores.items():
                        if score > 0:
                            identified_speaker = "Unknown"
                            # Try to load speaker's image
                        print(f"{_}: {score:.2f}")
                        #print(f"{speaker}: {score:.2f}")
                        tempscore += abs(score)
                    if tempscore > 200 :
                        identified_speaker = "Unknown" 
                    
                    speaker_image = self.load_and_resize_image(f"../images/{identified_speaker}.jpeg", (200, 200))
                    
                    if speaker_image:
                        self.root.after(0, lambda: self.image_label.configure(image=speaker_image))
                        self.speaker_image = speaker_image  # Keep a reference to prevent garbage collection

                    # Update GUI
                    self.root.after(0, self.update_result, identified_speaker)
                    
                except Exception as e:
                    self.root.after(0, messagebox.showerror, "Error", str(e))
                finally:
                    self.root.after(0, self.reset_interface)
            
            threading.Thread(target=process_recording).start()
    
    def update_result(self, speaker):
        result_text = f"I think you are...\n{speaker}!"
        self.result_label.config(text=result_text)
    
    def reset_interface(self):
        self.record_button.config(state='normal')
        self.status_label.config(text="Ready for another try!")
        self.progress.stop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeakerIdentificationGame(root)
    root.mainloop()