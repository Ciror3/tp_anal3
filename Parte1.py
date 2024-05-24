import numpy as np
import wave
import matplotlib.pyplot as plt
import os

# Ruta de la carpeta que contiene los archivos
carpeta = 'data/test'

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta)

def plot_waveform(signal, sample_freq, title):
    time = np.linspace(0, len(signal) / sample_freq, num=len(signal))
    plt.figure()
    plt.plot(time, signal)
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Amplitud')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Imprimir los nombres de los archivos
audios_opened = {}
sample_freq = {}
n_samples = {}
signal_wave = {}
audio_duration = {}
signal_array = {}
signal_fft = {}

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#Me dio cosa tocar el código, entonces hice este para la parte 1
i = 0
for nombre in archivos:
    for number in numbers:
        if nombre.startswith(number):
            audios_opened[nombre] = wave.open(os.path.join(carpeta, nombre), "rb")
            sample_freq[nombre] = audios_opened[nombre].getframerate()
            n_samples[nombre] = audios_opened[nombre].getnframes()
            signal_wave[nombre] = audios_opened[nombre].readframes(-1)
            audios_opened[nombre].close()
            audio_duration[nombre] = n_samples[nombre] / sample_freq[nombre]
            signal_array[nombre] = np.frombuffer(signal_wave[nombre], dtype=np.int16)
            signal_fft[nombre] = np.fft.fft(signal_array[nombre])  # Calcular la DFT de la señal

            #plot_waveform(signal_array[nombre], sample_freq[nombre], nombre)

