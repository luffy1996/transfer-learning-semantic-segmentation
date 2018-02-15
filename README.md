# Transfer Learning in Dilated Convolutional Network in Keras
This repository holds a Keras porting of the [ICLR 2016 paper](https://arxiv.org/abs/1511.07122) by Yu and Koltun and my proposed student teacher algorithm. 

## How to use

Please note that the porting works on with the Theano dim ordering.
Tensorflow backend should since if needed, the function `convert_all_kernels_in_model` is called.
However, it is not tested.

## Note
I tried to implement the student teacher hintloss method on voc dataset . How ever because of some technical difficulties , it is showing unsatisfactory results.