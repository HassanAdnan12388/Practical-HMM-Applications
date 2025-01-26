# Hidden-Markov-Models-Unraveling-Patterns-Across-Language-Music-and-Weather
HMM Fusion: Unraveling Patterns Across Language, Music, and Weather

# Hidden Markov Models for POS Tagging, Music Generation, and Weather Forecasting

## Project Overview

This project explores the application of Hidden Markov Models (HMM) across three different domains:

1. **Part of Speech (POS) Tagging:** Utilizing an HMM to predict POS tags for words in a given sentence using supervised learning and the Viterbi algorithm.
2. **Music Generation:** Implementing an unsupervised HMM with the Baum-Welch algorithm to learn patterns in classical music and generate new sequences.
3. **Weather Forecasting:** Leveraging the HMM to perform multi-step time series forecasting with covariates using the Meteostat dataset.

## Summary

This repository contains implementations for:
- **Supervised POS Tagging** using an HMM trained on the conll2000 dataset, with transition and emission probabilities estimated from labeled data.
- **Music Generation** through an HMM trained using the Baum-Welch algorithm on classical music data, producing melodies based on learned probabilities.
- **Weather Forecasting** using Gaussian HMMs to predict future temperature trends based on past data from Meteostat.

## Technologies Used

- **Python**
- **NumPy** for matrix operations
- **NLTK** for POS tagging datasets
- **MusPy** for symbolic music representation
- **HMMlearn** for HMM implementations
- **Meteostat** for retrieving weather data
- **Matplotlib** for visualization

## Architecture

### 1. POS Tagging (Supervised HMM)
- **Dataset:** conll2000 with universal_tagset
- **Training:** Estimation of transition and emission probabilities
- **Inference:** Viterbi algorithm for decoding
- **Evaluation:** Accuracy on a test set

### 2. Music Generation (Unsupervised HMM with Baum-Welch)
- **Dataset:** HaydnOp20 classical music dataset
- **Preprocessing:** Conversion to pitch representation
- **Training:** Baum-Welch algorithm for parameter estimation
- **Generation:** Sampling sequences using transition and emission probabilities

### 3. Weather Forecasting (HMM-based Time Series Prediction)
- **Dataset:** Temperature data from Meteostat for Lahore, PK
- **Training:** Gaussian HMM trained on historical temperature changes
- **Forecasting:** Predicting future temperatures using static and dynamic methods
- **Evaluation:** SMAPE loss for validation

## Results

- **POS Tagging:** Achieved high accuracy on unseen text using the trained HMM.
- **Music Generation:** Produced coherent melodies based on learned transitions.
- **Weather Forecasting:** Accurately predicted multi-step temperature trends with minimal error.


## Acknowledgments

- Stanford NLP Course Materials
- MusPy Documentation
- Meteostat API

## License

This project is licensed under the MIT License.
