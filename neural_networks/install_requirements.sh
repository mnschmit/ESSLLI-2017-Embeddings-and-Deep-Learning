#!/usr/bin/env bash

# install pytorch

if [[ "$1" == "conda" ]]
then
    echo "Installing pytorch using _conda_"
    conda install pytorch torchvision -c soumith
    # for CUDA 8.0
    # conda install cuda80 -c soumith
else
    echo "Installing pytorch using _pip_"

    # uncomment in case you do not have installed virtualenv yet
    # pip install virtualenv

    virtualenv --no-site-packages convolutional
    source convolutional/bin/activate

    # for CUDA 7.5 or none
    pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
    # for python 3.5
    # pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 

    # for CUDA 8.0
    # pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
    # for python 3.5
    # pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl


    if [[ $? -eq 0 ]]
    then
	pip install torchvision
    else
	deactivate
	exit 1
    fi
fi


# install torchtext

git clone https://github.com/pytorch/text.git
cd text

if [[ "$1" == "conda" ]]
then
    python setup.py build
    python setup.py install 
else
    pip install .

    deactivate

    echo "\n####\n"
    echo 'Type `source convolutional/bin/activate` in order to use the installed libraries.'
    echo 'After that type `deactivate` when you are done.'
fi
