#!/bin/bash -l
for f in ../converter/convert*.py;  do
python $f ;
echo $f
done