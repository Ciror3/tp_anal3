import numpy as np
import wave
import os
from scipy.stats import pearsonr

train = 'data/train'
test = 'data/test'
# Función para cargar y normalizar los datos de audio
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def load_and_normalize_audio_data(folder):
    archivos = os.listdir(folder)
    normalized_signal_dict = {number: [] for number in numbers}
    
    for nombre in archivos:
        for number in numbers:
            if nombre.startswith(number):
                # Leer y procesar el archivo de audio
                audios_opened = wave.open(os.path.join(folder, nombre), "rb")
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

# Función para hacer la suma punto a punto y calcular el promedio
def average_normalized_signals(normalized_signal_dict):
    averaged_signals = {}
    
    for number, signals in normalized_signal_dict.items():
        if signals:
            # Asegurarse de que todos los arrays tengan la misma longitud
            min_length = min(len(signal) for signal in signals)
            trimmed_signals = [signal[:min_length] for signal in signals]
            
            # Convertir la lista de arrays en un array 2D
            signals_array = np.array(trimmed_signals)
            
            # Calcular la suma punto a punto y luego el promedio
            sum_signals = np.sum(signals_array, axis=0)
            average_signal = sum_signals / len(signals)
            
            # Guardar el promedio en el diccionario
            averaged_signals[number] = average_signal
    
    return averaged_signals

# Función para comparar las señales normalizadas con el promedio
def compare_signals_with_averages(test_folder, averaged_signals):
    test_signal_dict = load_and_normalize_audio_data(test_folder)
    avgs = {number: [0 for _ in range(50)] for number in numbers}
    counter = 0
    for number, signals in test_signal_dict.items():
        if number in averaged_signals:
            average_signal = averaged_signals[number]
            for signal in signals:
                for i in range(0, len(signal)):
                    avgs[number][counter] += np.abs(signal[i]) - np.abs(average_signal[i])   
                avgs[number][counter] = avgs[number][counter]/len(average_signal)
                counter += 1
                if counter == 50:
                    counter = 0
    return avgs

def similarity_score(similarity_dict):
    similars = 0
    diff = 0

    for _, simil in similarity_dict.items():
        for i in range(50):
            if simil[i] < 0.05:
                similars += 1
            else:
                diff += 1 
    return similars, diff
            

normalized_signal_dict = load_and_normalize_audio_data(train)
# Calcular los promedios de las señales normalizadas
averaged_signals = average_normalized_signals(normalized_signal_dict)

similarity_results = compare_signals_with_averages(test, averaged_signals)
simil, diff = similarity_score(similarity_results)

prob_simil = 100*simil/500
prob_diff = 100*diff/500

print(f"Con un 95% de aprobacion, es igual en {prob_simil}% de las veces y distinto {prob_diff}% de las veces")

