# Scripts

We provide two scripts that can be run from command-line and perform some symmetry reconstruction task from start to finish. 
You can find both of them in the `src` directory.

### paper_exp.py

This scripts tries different ways to symmetrize a given graph; it can also be used to do the same job on a synthetic graph that the script itself produces.
Please run 

```
$ ./paper_exp.py --help
```

to have information on what the script does and how to run it.

### symmetrize.py

This scripts does the full symmetrization of a given graph. It produces a number of output files, stemmed from a specific basename; the content of the files is described in detail in a file called `basename-README.txt`, that the script itself produces
Please run 

```
$ ./symmetrize.py --help
```

to have information on what the script does and how to run it.