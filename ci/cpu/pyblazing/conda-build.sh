#!/bin/bash

#conda build -c blazingsql${NIGHTLY}/label/main/ -c rapidsai${NIGHTLY} -c conda-forge -c defaults --python=$PYTHON conda/recipes/pyBlazing/
conda build -c blazingsql/label/test/${NIGHTLY} -c rapidsai${NIGHTLY} -c conda-forge -c defaults --python=$PYTHON conda/recipes/pyBlazing/

