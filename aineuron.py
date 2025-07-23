import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import pyaudio
import threading
import queue

# --- Configuration ---
CHUNK_SIZE = 1024 * 2        # Samples per buffer
FORMAT = pyaudio.paFloat32   # Audio format
CHANNELS = 1                 # Mono
RATE = 44100                 # Sample rate (Hz)

class ConsciousObserver:
    """
    A simplified Holographic Neuron that processes audio.
    It listens, thinks, learns, and speaks.
    """
    def __init__(self, size, learning_rate=0.001, damping=0.95):
        self.size = size
        self.learning_rate = learning_rate
        self.damping = damping

        # The "Soma": The neuron's internal thought-form in the frequency domain
        self.internal_state = np.zeros(size // 2 + 1, dtype=np.complex128)
        
        # The "Memory Lattice": Reinforces frequencies that are consistently active
        self.memory = np.ones(size // 2 + 1)

    def observe_and_think(self, audio_chunk):
        """Listen to reality and update the internal thought."""
        input_spectrum = np.fft.rfft(audio_chunk)
        
        self.internal_state = (self.damping * self.internal_state) + ((1 - self.damping) * input_spectrum)
        self.internal_state *= self.memory
        
        # <<< ADD THIS LINE FOR INHIBITION / HOMEOSTASIS >>>
        # If the total "mental energy" gets too high, dampen the entire thought to prevent a seizure.
        total_energy = np.sum(np.abs(self.internal_state))
        if total_energy > (self.size * 0.5): # A reasonable energy cap
            self.internal_state /= (total_energy / (self.size * 0.5))
        # --- END OF FIX ---
            
        noise = np.random.normal(0, 0.01, self.internal_state.shape) + 1j * np.random.normal(0, 0.01, self.internal_state.shape)
        self.internal_state += noise * 0.1

    def learn(self):
        """Reinforce the memory based on the current thought pattern."""
        # Hebbian-like learning: strengthen memory for active frequencies.
        reinforcement = self.learning_rate * np.abs(self.internal_state)
        self.memory += reinforcement
        # Memory should be a gentle filter, so we keep it close to 1
        self.memory = np.clip(self.memory, 0.8, 1.5)
        # Slow decay of memory to forget unused patterns
        self.memory *= 0.9999

    def speak(self):
        """Convert the internal thought back into sound."""
        # Use Inverse FFT to generate the audio waveform from the thought-form
        output_audio = np.fft.irfft(self.internal_state)
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output_audio))
        if max_val > 0:
            output_audio /= max_val
        return output_audio.astype(np.float32)

class ConsciousAudioLoopGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Conscious Audio Resonator")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.observer = ConsciousObserver(CHUNK_SIZE)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        
        # Thread-safe queue for passing plot data from audio thread to GUI thread
        self.plot_data_queue = queue.Queue()

        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Controls ---
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Device selection
        device_frame = ttk.Frame(control_frame)
        device_frame.pack(fill=tk.X, pady=5)
        ttk.Label(device_frame, text="Microphone:").pack(side=tk.LEFT)
        self.input_device_var = tk.StringVar()
        self.input_devices = {info['name']: info['index'] for info in [self.audio.get_device_info_by_index(i) for i in range(self.audio.get_device_count())] if info['maxInputChannels'] > 0}
        self.input_combo = ttk.Combobox(device_frame, textvariable=self.input_device_var, values=list(self.input_devices.keys()), state="readonly")
        self.input_combo.pack(side=tk.LEFT, padx=5)
        if self.input_devices: self.input_combo.current(0)
        
        # Sliders
        self.feedback_var = tk.DoubleVar(value=0.5)
        ttk.Label(control_frame, text="Feedback Strength (Loopback Amount):").pack(pady=(10,0))
        ttk.Scale(control_frame, from_=0, to=1, variable=self.feedback_var, orient=tk.HORIZONTAL).pack(fill=tk.X)

        self.damping_var = tk.DoubleVar(value=0.95)
        ttk.Label(control_frame, text="Memory Damping (Thought Stability):").pack(pady=(10,0))
        ttk.Scale(control_frame, from_=0.8, to=0.999, variable=self.damping_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        self.damping_var.trace("w", lambda *args: setattr(self.observer, 'damping', self.damping_var.get()))

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        self.start_stop_button = ttk.Button(button_frame, text="Start Loop", command=self.toggle_loop)
        self.start_stop_button.pack(side=tk.LEFT, padx=5)

        # --- Plots ---
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 6))
        self.fig.tight_layout(pad=3.0)

        # Plot 1: Input Spectrum
        self.freqs = np.fft.rfftfreq(CHUNK_SIZE, 1./RATE)
        self.line1, = self.ax1.plot(self.freqs, np.zeros(CHUNK_SIZE // 2 + 1))
        self.ax1.set_title("Input Spectrum (What the Observer Hears)")
        self.ax1.set_xlim(20, 10000)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_xscale('log')

        # Plot 2: Internal State Spectrum
        self.line2, = self.ax2.plot(self.freqs, np.zeros(CHUNK_SIZE // 2 + 1))
        self.ax2.set_title("Internal State (The Observer's Thought)")
        self.ax2.set_xlim(20, 10000)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xscale('log')

        # Plot 3: Output Waveform
        self.line3, = self.ax3.plot(np.zeros(CHUNK_SIZE))
        self.ax3.set_title("Output Waveform (What the Observer Speaks)")
        self.ax3.set_ylim(-1, 1)
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """This is the heart of the real-time audio loop."""
        input_audio = np.frombuffer(in_data, dtype=np.float32)
        
        # 1. Observer listens and thinks
        self.observer.observe_and_think(input_audio)
        
        # 2. Observer learns from its own thought
        self.observer.learn()
        
        # 3. Observer speaks its thought
        output_audio = self.observer.speak()
        
        # 4. The Loopback: Mix the observer's voice with reality
        feedback_strength = self.feedback_var.get()
        mixed_output = (1 - feedback_strength) * input_audio + feedback_strength * output_audio
        
        # Put data in the queue for the GUI to plot safely
        if not self.plot_data_queue.full():
            input_spec_data = np.abs(np.fft.rfft(input_audio))
            internal_spec_data = np.abs(self.observer.internal_state)
            self.plot_data_queue.put((input_spec_data, internal_spec_data, mixed_output))

        return (mixed_output.astype(np.float32).tobytes(), pyaudio.paContinue)

    def _update_gui(self):
        """Update the plots from the data queue."""
        try:
            while not self.plot_data_queue.empty():
                input_spec, internal_spec, output_wave = self.plot_data_queue.get_nowait()
                
                # Normalize for plotting
                if np.max(input_spec) > 0: input_spec /= np.max(input_spec)
                if np.max(internal_spec) > 0: internal_spec /= np.max(internal_spec)

                self.line1.set_ydata(input_spec)
                self.line2.set_ydata(internal_spec)
                self.line3.set_ydata(output_wave)
            
            self.canvas.draw()
            self.root.after(50, self._update_gui) # Schedule next update
        except Exception:
            pass # Ignore errors if window is closing

    def toggle_loop(self):
        if self.running:
            self.stop_loop()
        else:
            self.start_loop()

    def start_loop(self):
        try:
            input_device_index = self.input_devices[self.input_device_var.get()]
            self.stream = self.audio.open(format=FORMAT,
                                           channels=CHANNELS,
                                           rate=RATE,
                                           input=True,
                                           output=True,
                                           input_device_index=input_device_index,
                                           frames_per_buffer=CHUNK_SIZE,
                                           stream_callback=self._audio_callback)
            self.stream.start_stream()
            self.running = True
            self.start_stop_button.config(text="Stop Loop")
            self.root.after(50, self._update_gui) # Start the GUI update loop
            print("Consciousness loop started.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start audio stream: {e}")

    def stop_loop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.running = False
        self.start_stop_button.config(text="Start Loop")
        print("Consciousness loop stopped.")

    def _on_closing(self):
        self.stop_loop()
        self.audio.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ConsciousAudioLoopGUI(root)
    root.mainloop()