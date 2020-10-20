# Music Machine Learning - ATIAM

## Tutorial 

<div markdown = "1">

In this introduction, we will cover basic Music Information Retrieval (MIR) interactions, in which we process a dataset of sound files and try to observe the properties of their various temporal and spectral features. Hence, we will quickly review basic calculus required to perform further machine learning tasks. This tutorial is also intended to review basic Matlab coding and plotting operations.

</div>{: .notice--blank}

## 0.0 - Reference code

<div markdown = "1">

Along the tutorials, we provide a reference code for each section. This code contains helper functions that will alleviate you from the burden of data import and other sideline implementations. You will find designated spaces in each file to develop your solutions.  **The newest version of this course is being actively developed as a set of Python Notebooks, that you can find at the following repo (that you should clone)**

[https://github.com/esling/atiam_ml](https://github.com/esling/atiam_ml)

**Please pull this repo regularly for the time of the course development to stay updated**

The code is in Python (notebooks impending) and relies heavily on the concept of [code sections](https://fr.mathworks.com/help/matlab/matlab_prog/run-sections-of-programs.html) which allows you to evaluate only part of the code (to avoid running long import tasks multiple times and concentrate on the question at hand.

Get the baseline MATLAB code for all tutorials from this [zip file ![](../images/file.png)](https://nuage.ircam.fr/index.php/s/F6QlLPgABOVJQRI)

Get the baseline Python code for all tutorials from this [zip file ![](../images/file.png)](../documents/Exercices_Python.zip)

Get the baseline Jupyter notebooks code for all tutorials from this [zip file ![](../images/file.png)](../documents/Exercices_Python.zip)

</div>{: .notice--blank}

### Dependencies

<div markdown = "1">

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
