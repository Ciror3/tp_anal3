import numpy as np
import wave
import os

train = 'data/train'
test = 'data/test'
# Función para cargar y normalizar los datos de audio
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def load_and_normalize_audio_data(folder, fourier):
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
                
                if fourier is True:
                    # Perform Fourier transform
                    fft_signal = np.fft.fft(signal_array)
                    fft_magnitude = np.abs(fft_signal)
                    max_val = np.max(fft_magnitude)
                    normalized_signal = fft_magnitude / max_val             
                    # Add the normalized signal to the corresponding dictionary
                    normalized_signal_dict[number].append(normalized_signal)
                else:
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
def compare_signals_with_averages(test_folder, averaged_signals,fourier):
    # Load and normalize the test signals
    test_signal_dict = load_and_normalize_audio_data(test_folder,fourier)
    
    results = {}
    count = 1
    # Compare each test signal with the corresponding average signal
    for number, test_signals in test_signal_dict.items():
        for _, test_signal in enumerate(test_signals):
            # Ensure the signals are of the same length for comparison
            for i in numbers:
                average_signal = averaged_signals[i]
                min_length = min(len(test_signal), len(average_signal))
                test_signal = test_signal[:min_length]
                average_signal = average_signal[:min_length]
                
                # Calculate the similarity (e.g., mean absolute error)
                error = np.mean(np.abs(test_signal - average_signal))
                
                # Store the result in a nested dictionary with unique identifiers for each test signal
                if f"test_signal_{count}_real_number_{number}" not in results:
                    results[f"test_signal_{count}_real_number_{number}"] = {}
                results[f"test_signal_{count}_real_number_{number}"][i] = error
            count += 1
    
    return results


def classifier(similarity_dict):
    top_three_results = {}
    
    for test_signal, errors in similarity_dict.items():
        # Sort the average signals by error in ascending order
        sorted_errors = sorted(errors.items(), key=lambda item: item[1])
        
        # Get the top three most similar average signals
        top_three = sorted_errors[:3]
        # Store the top three results in the new dictionary
        top_three_results[test_signal] = top_three
    
    return top_three_results

def acceptance_rate(tops):
    simmilarity = 0
    difference = 0
    simil_dict = {number:0 for number in numbers}
    for name, simils in tops.items():
        if name[-1] is simils[0][0]:
            simmilarity += 1
            simil_dict[name[-1]] += 1
        else:
            difference += 1
    return simmilarity,difference,simil_dict

print('Sin Fourier')
normalized_signal_dict = load_and_normalize_audio_data(train,False)
# Calcular los promedios de las señales normalizadas
averaged_signals = average_normalized_signals(normalized_signal_dict)
similarity_results = compare_signals_with_averages(test, averaged_signals,False)
tops = classifier(similarity_results)
simil, diff, simil_dict = acceptance_rate(tops)
print(f'Porcentaje de señales correctamente predichas: {(simil * 100) / 500}%, Porcentaje de señales incorrectamente predichas: {(diff * 100) / 500}%')
print(simil_dict)

print('Con Fourier')
normalized_signal_dict = load_and_normalize_audio_data(train,True)
averaged_signals = average_normalized_signals(normalized_signal_dict)
similarity_results = compare_signals_with_averages(test, averaged_signals,True)
tops = classifier(similarity_results)
simil, diff, simil_dict = acceptance_rate(tops)
print(f'Porcentaje de señales correctamente predichas: {(simil * 100) / 500}%, Porcentaje de señales incorrectamente predichas: {(diff * 100) / 500}%')
print(simil_dict)