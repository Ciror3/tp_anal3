import numpy as np
import wave
import os

# Ruta de la carpeta que contiene los archivos
carpeta = 'data/test'

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta)

# Lista de números del 0 al 9
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def load_audio_data():
    """Carga los datos de los archivos de audio y normaliza las señales."""
    # Diccionario para almacenar las señales de audio normalizadas por número
    normalized_signal_dict = {number: [] for number in numbers}
    
    for nombre in archivos:
        for number in numbers:
            if nombre.startswith(number):
                # Leer y procesar el archivo de audio
                audios_opened = wave.open(os.path.join(carpeta, nombre), "rb")
                sample_freq = audios_opened.getframerate()
                n_samples = audios_opened.getnframes()
                signal_wave = audios_opened.readframes(-1)
                audios_opened.close()
                audio_duration = n_samples / sample_freq
                signal_array = np.frombuffer(signal_wave, dtype=np.int16)
                
                # Normalizar la señal
                max_val = np.max(np.abs(signal_array))
                normalized_signal = signal_array / max_val
                
                # Agregar la señal normalizada al diccionario correspondiente
                normalized_signal_dict[number].append(normalized_signal)
    
    return normalized_signal_dict

def calculate_mean(normalized_signal_dict):
    """Calcula la media para cada número a partir de los datos normalizados."""
    # Diccionario para almacenar la media de cada número
    mean_dict = {}
    
    for number, signals in normalized_signal_dict.items():
        # Concatenar todas las señales normalizadas del número
        concatenated_signals = np.concatenate(signals)
        
        # Calcular la media de las señales concatenadas
        mean_value = np.mean(concatenated_signals)
        
        # Almacenar la media en el diccionario
        mean_dict[number] = mean_value
    
    return mean_dict

# Cargar los datos de audio
normalized_signal_dict = load_audio_data()

# Calcular la media de cada número
mean_dict = calculate_mean(normalized_signal_dict)

# Imprimir la media de cada número
print("Media de cada número:")
for number, mean_value in mean_dict.items():
    print(f"Número {number}: {mean_value}")


