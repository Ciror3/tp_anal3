import numpy as np
import wave
import os

train = 'data/train'
test = 'data/test'
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Function to load and normalize audio data
def load_and_normalize_audio_data(folder):
    archivos = os.listdir(folder)
    normalized_signal_dict = {number: [] for number in numbers}
    
    for nombre in archivos:
        for number in numbers:
            if nombre.startswith(number):
                # Read and process the audio file
                audios_opened = wave.open(os.path.join(folder, nombre), "rb")
                sample_freq = audios_opened.getframerate()
                n_samples = audios_opened.getnframes()
                signal_wave = audios_opened.readframes(-1)
                audios_opened.close()
                signal_array = np.frombuffer(signal_wave, dtype=np.int16)
                
                # Perform Fourier transform
                fft_signal = np.fft.fft(signal_array)
                fft_magnitude = np.abs(fft_signal)
                
                # Normalize the signal
                max_val = np.max(fft_magnitude)
                normalized_signal = fft_magnitude / max_val
                
                # Add the normalized signal to the corresponding dictionary
                normalized_signal_dict[number].append(normalized_signal)
    
    return normalized_signal_dict

# Function to average normalized signals
def average_normalized_signals(normalized_signal_dict):
    averaged_signals = {}
    
    for number, signals in normalized_signal_dict.items():
        if signals:
            # Ensure all arrays have the same length
            min_length = min(len(signal) for signal in signals)
            trimmed_signals = [signal[:min_length] for signal in signals]
            
            # Convert the list of arrays into a 2D array
            signals_array = np.array(trimmed_signals)
            
            # Calculate the point-by-point sum and then the average
            sum_signals = np.sum(signals_array, axis=0)
            average_signal = sum_signals / len(signals)
            
            # Store the average in the dictionary
            averaged_signals[number] = average_signal
    
    return averaged_signals

# Function to compare normalized signals with averages
def compare_signals_with_averages(test_folder, averaged_signals):
    test_signal_dict = load_and_normalize_audio_data(test_folder)
    
    results = {}
    count = 1
    for number, test_signals in test_signal_dict.items():
        for _, test_signal in enumerate(test_signals):
            for i in numbers:
                average_signal = averaged_signals[i]
                min_length = min(len(test_signal), len(average_signal))
                test_signal = test_signal[:min_length]
                average_signal = average_signal[:min_length]
                
                # Calculate similarity (e.g., mean absolute error)
                error = np.mean(np.abs(test_signal - average_signal))
                
                if f"test_signal_{count}_real_number_{number}" not in results:
                    results[f"test_signal_{count}_real_number_{number}"] = {}
                results[f"test_signal_{count}_real_number_{number}"][i] = error
            count += 1
    
    return results

# Function to classify the signals based on similarity
def classifier(similarity_dict):
    top_three_results = {}
    
    for test_signal, errors in similarity_dict.items():
        sorted_errors = sorted(errors.items(), key=lambda item: item[1])
        top_three = sorted_errors[:3]
        top_three_results[test_signal] = top_three
    
    return top_three_results

# Function to calculate acceptance rate
def acceptance_rate(tops):
    similarity = 0
    difference = 0
    simil_dict = {number:0 for number in numbers}
    for name, simils in tops.items():
        if name[-1] == simils[0][0]:
            similarity += 1
            simil_dict[name[-1]] += 1
        else:
            difference += 1
    return similarity, difference, simil_dict

# Main process
normalized_signal_dict = load_and_normalize_audio_data(train)
averaged_signals = average_normalized_signals(normalized_signal_dict)
similarity_results = compare_signals_with_averages(test, averaged_signals)
tops = classifier(similarity_results)
simil, diff, simil_dict = acceptance_rate(tops)
print(f'Percentage of correctly predicted signals: {(simil * 100) / 500}%, Percentage of incorrectly predicted signals: {(diff * 100) / 500}%')
print(simil_dict)
