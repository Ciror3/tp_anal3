import numpy as np
import wave
import librosa
import os
import scipy.signal as sig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

train = 'data/train'
test = 'data/test'
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#Funcion no utilizada
def correlation_lags(in1_len, in2_len, mode='full'):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.
    Parameters
    ----------
    in1_size : int
        First input size.
    in2_size : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.
    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.
    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.
    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:
    .. math ::
        \left ( f\star g \right )\left ( \tau \right )
        \triangleq \int_{t_0}^{t_0 +T}
        \overline{f\left ( t \right )}g\left ( t+\tau \right )dt
    Where :math:`\tau` is defined as the displacement, also known as the lag.
    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:
    .. math ::
        \left ( f\star g \right )\left [ n \right ]
        \triangleq \sum_{-\infty}^{\infty}
        \overline{f\left [ m \right ]}g\left [ m+n \right ]
    Where :math:`n` is the lag.
    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.
    >>> from scipy import signal
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = np.concatenate([rng.standard_normal(100), x])
    >>> correlation = signal.correlate(x, y, mode="full")
    >>> lags = signal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[np.argmax(correlation)]
    """

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags

#Funcion para cargar el audio y preprocesar los datos
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

#Encontrar las señal promedio de una cantidad N de funciones
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

#Funcion para comparar señales de test con los promedios conseguidos del entrenamiento
def compare_signals_with_averages(test_folder, averaged_signals, mode):
    test_signal_dict = load_audio_data(test_folder, mode)
    results = {}
    count = 1
    
    for number, test_signals in test_signal_dict.items():
        for test_signal in test_signals:
            for i in numbers:
                
                """
                Intento de correlacion
                """
                # correlation = sig.correlate(test_signal, average_signal, mode='full')
                # lags = correlation_lags(len(test_signal), len(average_signal))
                # optimal_lag = lags[np.argmax(correlation)]
                
                # 
                # if optimal_lag > 0:
                #     aligned_test_signal = test_signal[optimal_lag:]
                #     aligned_average_signal = average_signal[:len(aligned_test_signal)]
                # else:
                #     aligned_average_signal = average_signal[-optimal_lag:]
                #     aligned_test_signal = test_signal[:len(aligned_average_signal)]
                average_signal = averaged_signals[i]
                min_length = min(len(test_signal), len(average_signal))
                test_signal = test_signal[:min_length]
                average_signal = average_signal[:min_length]
                
                error = np.mean((test_signal - average_signal) ** 2)
                                
                key = f"test_signal_{count}_real_number_{number}"
                if key not in results:
                    results[key] = {}
                results[key][i] = error
            count += 1
    
    return results

#Clasificador para cada señal de prueba
def classifier(similarity_dict):
    predictions = {}
    for test_signal, errors in similarity_dict.items():
        sorted_errors = sorted(errors.items(), key=lambda item: item[1])
        predicted_label = sorted_errors[0][0]
        true_label = test_signal.split('_')[-1]
        predictions[test_signal] = (true_label, predicted_label)

    return predictions

#Calcular los porcentajes de aciertos del clasificador
def acceptance_rate(predictions):
    correct = 0
    total = len(predictions)
    simil_dict = {number: 0 for number in numbers}
    for _, (true_label, predicted_label) in predictions.items():
        if true_label == predicted_label:
            correct += 1
            simil_dict[true_label] += 1

    return correct, total - correct, simil_dict

#Generador de matrices de confusion
def generate_confusion_matrix(predictions):
    true_labels = [true_label for true_label, _ in predictions.values()]
    predicted_labels = [predicted_label for _, predicted_label in predictions.values()]
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=numbers)
    class_report = classification_report(true_labels, predicted_labels, labels=numbers, target_names=numbers)
    return conf_matrix, class_report, true_labels, predicted_labels

#Aplicacion del codigo para cada opcion de preprocesamiento
modes = ['Sin Fourier', 'Fourier', 'MFCC']
for mode in modes:
    print(f'\n{mode}')
    normalized_signal_dict = load_audio_data(train, mode.lower())
    averaged_signals = average_signals(normalized_signal_dict)
    similarity_results = compare_signals_with_averages(test, averaged_signals, mode.lower().replace(' ', ''))
    predictions = classifier(similarity_results)
    simil, diff, simil_dict = acceptance_rate(predictions)
    conf_matrix, class_report, true_labels, predicted_labels = generate_confusion_matrix(predictions)
    print(f'Porcentaje de señales correctamente predichas: {(simil * 100) / 500}%, Porcentaje de señales incorrectamente predichas: {(diff * 100) / 500}%')
    mat = ConfusionMatrixDisplay(confusion_matrix(true_labels, predicted_labels, labels=numbers), display_labels=numbers)
    mat.plot()
    plt.xlabel('Etiqueta Predicha') 
    plt.ylabel('Etiqueta Verdadera') 
    plt.show()


