# DeepECG
ECG classification programs based on ML/DL methods. There are two datasets:
 - **training2017.zip** file contains one electrode voltage measurements taken as the difference between RA and LA electrodes with no ground. It is taken from The 2017 PhysioNet/CinC Challenge.
 - **MIT-BH.zip file** contains two electrode voltage measurements: MLII and V5.

## Prerequisites:
- Python 3.5 and higher
- Keras framework with TensorFlow backend
- Numpy, Scipy, Pandas libs
- Scikit-learn framework 

## Instructions for running the program
1) Execute the **training2017.zip** and **MIT-BH.zip** files into folders **training2017/** and **MIT-BH/** respectively
2) If you want to use 2D Convolutional Neural Network for ECG classification then run the file **CNN_ECG.py** with the following commands:
 - If you want to train your model on the 2017 PhysioNet/CinC Challenge dataset:
```
python CNN_ECG.py cinc
```
 - If you want to train your model on the MIT-BH dataset:
```
python CNN_ECG.py mit
```
3) If you want to use 1D Convolutional Neural Network for ECG classification then run the file **Conv1D_ECG.py** with the following commands:
```
python Conv1D_ECG.py
```
  
# Additional info
### Citation
If you use my repo - then, please, cite my paper. This is a BibTex citation:


    @article{pyakillya_kazachenko_mikhailovsky_2017,
        author = {Boris Pyakillya, Natasha Kazachenko, Nick Mikhailovsky},
        title = {Deep Learning for ECG Classification},
        journal = {Journal of Physics: Conference Series},
        year = {2017},
        volume = {913},
        pages = {1-5},
        DOI={10.1088/1742-6596/913/1/012004},
        url = {http://iopscience.iop.org/article/10.1088/1742-6596/913/1/012004/pdf}
    }


### For feature extraction and hearbeat rate calculation:
- https://github.com/PIA-Group/BioSPPy (Biosignal Processing in Python)
