# Detection of Postpartum Depression Using EEG and Deep Learning

An end-to-end signal processing and deep learning pipeline for multi-class classification of Postpartum Depression (PPD) severity using 7-channel prefrontal Electroencephalogram (EEG) signals. The pipeline converts inter-channel spectral power differences in the Theta and Beta frequency bands into 2D Brain Asymmetry Matrix images and classifies them into three severity levels using a lightweight 2D Convolutional Neural Network (2D-CNN).

## Dataset Overview

The model is trained on 7-channel prefrontal EEG recordings (`Fp1`, `F3`, `F7`, `Fz`, `Fp2`, `F4`, `F8`) stored in `.edf` (European Data Format) and labeled using Edinburgh Postnatal Depression Scale (EPDS) scores across three classes:

* **Class 1**: Healthy controls
* **Class 2**: Mildly / Moderately depressed
* **Class 3**: Severely depressed

## Pipeline & Methodology

1. **EEG Preprocessing** (MNE-Python and scikit-learn):
   * Channel extraction and standard 10–20 montage mapping.
   * Bandpass filtering (0.1–70 Hz) and power-line notch filtering (50 Hz).
   * Channel-wise Min-Max normalization (`MinMaxScaler`) and artifact removal via Independent Component Analysis (`FastICA`).
2. **Feature Extraction** (SciPy, Matplotlib and OpenCV):
   * Power Spectral Density (PSD) estimation using Welch's periodogram across Theta (4–8 Hz), Alpha (8–12 Hz), and Beta (12–30 Hz) bands.
   * Generation of 7 × 7 inter-channel relative power asymmetry heatmaps resized to 64 × 64 × 3 RGB images for the Theta and Beta bands.
3. **Deep Learning Classification** (TensorFlow/Keras):
   * **Proposed 2D-CNN**: A 4-block sequential CNN (`Conv2D` with 32, 64, 128, 256 filters + `MaxPooling2D`, followed by `Flatten`, `Dense(512)`, `Dropout(0.25)`, and a 3-class `Softmax` output layer) trained with the `Adam` optimizer and categorical cross-entropy loss.
   * **Transfer Learning Baselines**: Pre-trained [VGG-16](https://keras.io/api/applications/vgg/vgg_models/#vgg16-function), [VGG-19](https://keras.io/api/applications/vgg/vgg_models/#vgg19-function), and [Xception](https://keras.io/api/applications/xception/) models evaluated on the same 3-class task.

## Results

| EEG Frequency Band | VGG-16 | VGG-19 | Xception | Proposed 2D-CNN |
| :--- | :---: | :---: | :---: | :---: |
| **Theta Wave (4–8 Hz)** | 87.5% | 75.0% | 75.0% | **87.5%** |
| **Beta Wave (12–30 Hz)** | 75.0% | 75.0% | 75.0% | **75.0%** |
