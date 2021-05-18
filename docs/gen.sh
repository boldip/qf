#!/bin/bash
yes | pip uninstall qf
cd ..
python setup.py install
cd -
make clean html
