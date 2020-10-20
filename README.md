# Music Machine Learning - ATIAM

This repository contains the most up to date courses in machine learning applied to music computing given along the ATIAM Masters at IRCAM. The courses slides along with a set of interactive Jupyter Notebooks will be updated along the year to provide all the ML program.

**As the development of this course is ongoing, please pull this repo regularly to stay updated**

Please first follow the installation procedure (see next section) to ensure that you have all necessary libraries to follow the course smoothly. You also need to get the audio datasets from this [link ![](../images/file.png)](https://nuage.ircam.fr/index.php/s/FTsaaAMFV1jEwsk)   

## Installation and dependencies


Along the tutorials, we provide a reference code for each section. This code contains helper functions that will alleviate you from the burden of data import and other sideline implementations. You will find designated spaces in each file to develop your solutions. The code is in Python (notebooks impending) and relies heavily on the concept of [code sections](https://fr.mathworks.com/help/matlab/matlab_prog/run-sections-of-programs.html) which allows you to evaluate only part of the code (to avoid running long import tasks multiple times and concentrate on the question at hand.

### Dependencies

#### Python installation

In order to get the baseline script to work, you need to have a working distribution of `Python 3.5` as a minimum (we also recommend to update your version to `Python 3.7`). We will also be using the following libraries

- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Music21](http://web.mit.edu/music21/)
- [Librosa](http://librosa.github.io/librosa/index.html)
- [PyTorch](https://pytorch.org/)

We highly recommend that you install [Pip](https://pypi.python.org/pypi/pip/) or [Anaconda](https://www.anaconda.com/download/) that will manage the automatic installation of those Python libraries (along with their dependencies). If you are using `Pip`, you can use the following commands

```
pip install matplotlib
pip install numpy
pip install scipy
pip install scikit-learn
pip install music21
pip install librosa
pip install torch torchvision
```

For those of you who have never coded in Python, here are a few interesting resources to get started.

- [TutorialPoint](https://www.tutorialspoint.com/python/)
- [Programiz](https://www.programiz.com/python-programming)

#### Jupyter notebooks and lab

In order to ease following the exercises along with the course, we will be relying on [**Jupyter Notebooks**](https://jupyter.org/). If you have never used a notebook before, we recommend that you look at their website to understand the concept. Here we also provide the instructions to install **Jupyter Lab** which is a more integrative version of notebooks. You can install it on your computer as follows (if you use `pip`)

```
pip install jupyterlab
```

Then, once installed, you can go to the folder where you cloned this repository, and type in

```
jupyter lab
```
