import numpy as np
import wave
import librosa
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

train = 'data/train'
test = 'data/test'
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def load_audio_data(folder, mode):
    archivos = os.listdir(folder)
    signal_dict = {number: [] for number in numbers}
    
    for nombre in archivos:
        for number in numbers:
            if nombre.startswith(number):
                audios_opened = wave.open(os.path.join(folder, nombre), "rb")
                sample_rate = audios_opened.getframerate()
                signal_wave = audios_opened.readframes(-1)
                audios_opened.close()
                signal_array = np.frombuffer(signal_wave, dtype=np.int16)
                
                if mode == 'fourier':
                    fft_signal = np.fft.fft(signal_array)
                    fft_magnitude = np.abs(fft_signal)
                    signal_dict[number].append(fft_magnitude)
                elif mode == 'mfcc':
                    mfccs = librosa.feature.mfcc(y=signal_array.astype(float), sr=sample_rate, n_mfcc=13)
                    signal_dict[number].append(mfccs.mean(axis=1))
                else:
                    signal_dict[number].append(signal_array)
    
    return signal_dict

def average_signals(signal_dict):
    averaged_signals = {}
    
    for number, signals in signal_dict.items():
        if signals:
            min_length = min(len(signal) for signal in signals)
            trimmed_signals = [signal[:min_length] for signal in signals]
            signals_array = np.array(trimmed_signals)
            sum_signals = np.sum(signals_array, axis=0)
            average_signal = sum_signals / len(signals)
            averaged_signals[number] = average_signal
    
    return averaged_signals

def compare_signals_with_averages(test_folder, averaged_signals, fourier):
    test_signal_dict = load_audio_data(test_folder, fourier)
    
    results = {}
    count = 1
    for number, test_signals in test_signal_dict.items():
        for test_signal in test_signals:
            for i in numbers:
                average_signal = averaged_signals[i]
                min_length = min(len(test_signal), len(average_signal))
                test_signal = test_signal[:min_length]
                average_signal = average_signal[:min_length]
                error = np.mean(np.abs((test_signal - average_signal)))
                
                if f"test_signal_{count}_real_number_{number}" not in results:
                    results[f"test_signal_{count}_real_number_{number}"] = {}
                results[f"test_signal_{count}_real_number_{number}"][i] = error
            count += 1
    
    return results

def classifier(similarity_dict):
    predictions = {}
    
    for test_signal, errors in similarity_dict.items():
        sorted_errors = sorted(errors.items(), key=lambda item: item[1])
        predicted_label = sorted_errors[0][0]
        true_label = test_signal.split('_')[-1]
        predictions[test_signal] = (true_label, predicted_label)
    
    return predictions

def acceptance_rate(predictions):
    correct = 0
    total = len(predictions)
    simil_dict = {number: 0 for number in numbers}
    
    for _, (true_label, predicted_label) in predictions.items():
        if true_label == predicted_label:
            correct += 1
            simil_dict[true_label] += 1
    
    return correct, total - correct, simil_dict

def generate_confusion_matrix(predictions):
    true_labels = [true_label for true_label, _ in predictions.values()]
    predicted_labels = [predicted_label for _, predicted_label in predictions.values()]
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=numbers)
    class_report = classification_report(true_labels, predicted_labels, labels=numbers, target_names=numbers)
    return conf_matrix, class_report, true_labels, predicted_labels

modes = ['Sin Fourier', 'fourier', 'mfcc']
for mode in modes:
    print(f'\n{mode}')
    normalized_signal_dict = load_audio_data(train, mode)
    averaged_signals = average_signals(normalized_signal_dict)
    similarity_results = compare_signals_with_averages(test, averaged_signals, mode.lower().replace(' ', ''))
    predictions = classifier(similarity_results)
    simil, diff, simil_dict = acceptance_rate(predictions)
    conf_matrix, class_report, true_labels, predicted_labels = generate_confusion_matrix(predictions)
    print(f'Porcentaje de señales correctamente predichas: {(simil * 100) / 500}%, Porcentaje de señales incorrectamente predichas: {(diff * 100) / 500}%')
    ConfusionMatrixDisplay(confusion_matrix(true_labels, predicted_labels, labels=numbers), display_labels=numbers).plot()
    plt.show()