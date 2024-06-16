import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize

# Rutas a los archivos de predicciones
pred_path_Raw = 'predictions/AudioMNIST_raw_waveform/test_predictions.npy' #19.40%
pred_path_Mfcc = 'predictions/AudioMNIST_mfcc/test_predictions.npy' #16.60%
pred_path_Dft = 'predictions/AudioMNIST_audio_spectrum/test_predictions.npy' #91.20% 

predictions_paths = [pred_path_Mfcc, pred_path_Raw, pred_path_Dft]

# Procesar cada archivo de predicciones
for prediction_path in predictions_paths:
    # Cargar las predicciones
    data = np.load(prediction_path, allow_pickle=True)
    input_data = data['i']
    true_labels = data['o']
    predictions = data['p']

    # Convertir a arrays
    input_data = np.concatenate(input_data, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    # Calcular la precisión
    if true_labels.ndim == 1:
        correct_predictions = np.sum(np.argmax(predictions, axis=1) == true_labels)
    else:
        correct_predictions = np.sum(np.argmax(predictions, axis=1) == np.argmax(true_labels, axis=1))

    total_predictions = len(true_labels)
    accuracy_viejo = (correct_predictions / total_predictions) * 100
    print(f'Accuracy Viejo: {accuracy_viejo:.2f}%')

    # Convertir las predicciones a etiquetas si son probabilidades
    if predictions.ndim > 1:
        predicted_labels = np.argmax(predictions, axis=1)
    else:
        predicted_labels = predictions

    # Calcular y mostrar la precisión y el informe de clasificación
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'Accuracy: {accuracy:.2f}%')
    print(classification_report(true_labels, predicted_labels))

    # Calcular y visualizar la matriz de confusión
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # Binarizar las etiquetas verdaderas
    true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = true_labels_bin.shape[1]

    # Calcular la curva ROC y el área bajo la curva para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcular la curva ROC y el área bajo la curva para la media micro
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_bin.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Visualizar la curva ROC
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]), alpha=0.3)  # Añadido alpha=0.3

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()