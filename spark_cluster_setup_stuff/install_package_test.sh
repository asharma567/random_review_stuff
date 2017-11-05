#!/bin/sh
# filename: bootstrap-simplecv.sh  (save it in an S3 bucket)
set -e -x

sudo apt-get install python-setuptools
sudo easy_install pip 
sudo pip install -U SimpleCV
sudo pip install -U py-stringmatching

