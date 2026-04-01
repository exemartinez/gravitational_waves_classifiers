# gravitational_waves_classifiers

This repository contains the code, trained artifacts, and final report for my ITBA Master's Degree capstone in Data Science. In this project, I studied the binary classification of Gravity Spy spectrograms to identify gravitational-wave `Chirp` events against non-`Chirp` noise classes, comparing deep learning models with classical machine learning baselines.

## Project Summary

The core idea behind the project is straightforward: take labeled spectrograms from Gravity Spy, turn them into a consistent numerical representation, and benchmark several binary classifiers on the task of identifying `Chirp` signals.

The pipeline I implemented is:

1. Read spectrograms and metadata from the original Gravity Spy HDF5 and CSV files.
2. Build a consolidated dataframe with the image matrix for each observation.
3. Convert the original multiclass labels into a binary target:
   - `1`: `Chirp`
   - `0`: any other class
4. Export normalized train, validation, and test datasets.
5. Train several model families and compare their predictive behavior.
6. Save trained models, learning curves, ROC plots, and predicted positives for manual inspection.

## Highlights

- End-to-end preprocessing pipeline from raw Gravity Spy files to train/validation/test tensors
- Binary classification framing focused on `Chirp` detection
- Comparative experimentation across CNNs, RNNs, SVMs, and LightGBM
- Persisted trained models and evaluation plots included in the repository
- Final academic report included as supporting documentation

## Models Explored

I implemented and compared the following model families:

- Convolutional neural networks for direct image-based classification
- Recurrent neural networks using LSTM, GRU, and SimpleRNN variants
- Linear SVM classifiers
- LightGBM classifiers

For the classical models, spectrograms are flattened into vectors of length `140 * 170 = 23800`. For the deep learning models, the spectrograms are preserved as image tensors.

## Repository Structure

```text
.
|-- deep/
|   |-- deep_environment.yml         Original Conda environment used for the experiments
|   `-- create_conda_env.sh          Helper script for Conda environment creation
|-- gw/
|   |-- generate_gw_dataset.py       Builds the consolidated dataset from Gravity Spy files
|   |-- normalize_dataset.py         Exports train/validation/test NumPy arrays
|   |-- train_*.py                   Training scripts for each model family
|   |-- predict_*.py                 Evaluation, ROC generation, and positive-sample export
|   `-- dataset_image_analysis.py    Descriptive analysis of the spectrogram dataset
|-- models/                          Saved trained models
|-- models_images/                   ROC curves, accuracy curves, and loss plots
|-- Chirps.pdf                       Supporting reference material
`-- TFI Hernan Ezequiel Martinez.pdf Final written report
```

## Dataset

The repository does not include the original Gravity Spy dataset. To reproduce the experiments, the following files are expected under a local `dataset/` directory:

```text
dataset/
|-- trainingsetv1d1.h5
|-- trainingset_v1d1_metadata.csv
`-- gw_images/
```

The preprocessing scripts generate these derived files:

```text
dataset/
|-- gw_consolidated_matrix.pickle
|-- gw_train_images.npy
|-- gw_train_labels.npy
|-- gw_validation_images.npy
|-- gw_validation_labels.npy
|-- gw_test_images.npy
`-- gw_test_labels.npy
```

I do not commit the dataset itself to the repository.

## Installation

The project was developed with Python `3.7.6`. I added a pinned `requirements.txt` to make the Python environment reproducible.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer Conda, the original environment is also available in `deep/deep_environment.yml`.

## Reproducing the Pipeline

### 1. Build the consolidated dataset

```bash
python gw/generate_gw_dataset.py
```

This script reads the original Gravity Spy files and stores a pandas dataframe containing both metadata and the spectrogram matrix for each sample.

### 2. Normalize and export train/validation/test arrays

```bash
python gw/normalize_dataset.py
```

This step converts the stored matrices into NumPy arrays used by the training scripts.

### 3. Train a model

Examples:

```bash
python gw/train_covnet_model.py
python gw/train_linear_svc_model.py
python gw/train_gbm_model.py
python gw/train_revnet_lstm_model.py
```

Additional variants are included for balancing and resampling experiments:

- `train_gbm_model_balanced.py`
- `train_gbm_model_balanced_weights.py`
- `train_gbm_model_exploded_weights.py`
- `train_linear_svc_reduced_model.py`
- `train_linear_svc_reduced_model_custom_set.py`
- `train_revnet_lstm_balanced_model.py`
- `train_revnet_lstm_exploted_model.py`
- `train_revnet_gru_dropout_model.py`
- `train_revnet_simplernn.py`

### 4. Evaluate a trained model

Examples:

```bash
python gw/predict_convnet_gw.py
python gw/predict_linearsvc_gw.py
python gw/predict_gbm_gw.py
python gw/predict_revnet_lstm_gw.py
```

These evaluation scripts typically:

- score the validation and test sets,
- compute ROC and AUC metrics,
- generate plots in `models_images/`,
- export predicted positive spectrograms to `dataset/gw_images/`.

## Main Scripts

### Data preparation

- `gw/generate_gw_dataset.py`: loads the source dataset and consolidates metadata plus spectrogram matrices into a pickle file.
- `gw/normalize_dataset.py`: converts the consolidated dataset into model-ready NumPy arrays.
- `gw/dataset_image_analysis.py`: computes descriptive statistics over the spectrogram set.

### Deep learning

- `gw/train_covnet_model.py`: convolutional baseline for spectrogram classification.
- `gw/train_revnet_lstm_model.py`: LSTM-based recurrent baseline.
- `gw/train_revnet_gru_dropout_model.py`: GRU-based recurrent baseline.
- `gw/train_revnet_simplernn.py`: SimpleRNN baseline.

### Classical machine learning

- `gw/train_linear_svc_model.py`: linear SVM with class balancing.
- `gw/train_linear_svc_reduced_model.py`: linear SVM trained on a reduced balanced subset.
- `gw/train_gbm_model.py`: LightGBM baseline.
- `gw/train_gbm_model_balanced_weights.py`: LightGBM using balanced class weights.

## Included Artifacts

This repository also includes artifacts generated during the project:

- trained models under `models/`
- ROC curves and training plots under `models_images/`
- the final report in PDF format

These files make the repository useful not only as source code, but also as a compact record of the experimental results.

## Notes

- The codebase is organized as independent scripts rather than as a packaged library.
- Paths are defined relative to the repository root.
- The repository reflects the research workflow used in the project and preserves the original experimentation style.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Contact

If you are reviewing this project in the context of research, data science, or machine learning engineering roles, feel free to reach out through my GitHub profile.
