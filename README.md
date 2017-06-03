# DeepECG
ECG classification programs based on ML/DL methods. Training ECG data (training2017/) consists one electrode voltage measurements taken as the difference between RA and LA electrodes with no ground.

## Prerequisites:
- Python 3.5 and higher
- Keras framework with TensorFlow backend
- Numpy, Scipy, Pandas libs
- Scikit-learn framework 

## Instructions for running the program
1) Execute the training2017.zip file into folder **training2017/**
2) Run the file CNN_ECG.py with the following command:
  
```
python3 ECG_CNN.py
```

# Additional info
### For feature extraction and hearbeat rate calculation:
- https://github.com/PIA-Group/BioSPPy (Biosignal Processing in Python)
