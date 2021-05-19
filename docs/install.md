# Installation

We recommend using the latest version of Python 3, qf supports Python 3.7 and newer. 
We also suggest that you create and install qf under a virtual environment. The following installation steps assume that you want to use pipenv.

### Cloning qf

Clone the project with the command:

```
$ git clone https://github.com/boldip/qf.git
```

### Creating the virtual environment

Enter in the project's directory and create a virtual environment.

```
$ cd qf
$ pipenv install
```

If you are a MacOS user, we suggest that you select specific Python version prior to 3.9 when you create the environment, issuing the command

```
$ pipenv install —python 3.7
```

Of course, you must have that version of Python installed in your system.

### Entering the virtual environment

Now, when in the directory of the project you can enter the virtual environment with the command.

```
$ pipenv shell
```

and issue the command `exit` to get out of the virtual environment.

### Installing the library

Enter the virtual environment and then issue the command

```
$ pip install .
```

to install the library. It will, in turn, install all the dependencies. This operation needs to be performed only once.

### Check that everything works

To check that everything works, move to the `src` directory and try to execute a simulation on a synthetic graph:

```
$ cd src
$ ./paper_exp.py :20:5 --minutes 1 --depth 2
```

### Installing a kernel to use jupyter

If you want to use jupyter with all the dependencies required by the library, you way connect the environment to a kernel

```
$ python -m ipykernel install --user --display-name qf --name qf
```

Now, to try out this kernel, cd to the src directory (if you are not already there) and
start the tutorial:

```
$ cd src
$ jupyter notebook tutorial.ipynb
```

Upon entering the notebook, check that the Kernel `qf` is correctly selected.
