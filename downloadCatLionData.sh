#! /bin/bash
cd data
wget http://people.csail.mit.edu/sumner/research/deftransfer/data/cat-poses.zip
unzip cat-poses.zip
mv cat-poses/cat-reference.obj cat.obj

wget http://people.csail.mit.edu/sumner/research/deftransfer/data/lion-poses.zip
unzip lion-poses.zip
mv lion-poses/lion-reference.obj lion.obj
