# DeepECG
ECG classification programs based on ML/DL methods. There are two datasets:
 - training2017.zip file contains one electrode voltage measurements taken as the difference between RA and LA electrodes with no ground. It is taken from The 2017 PhysioNet/CinC Challenge.
 - MIT-BH.zip file contains two electrode voltage measurements: MLII and V5.

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
