# DeepStackedLearner
Deep learning without back propagation.  Stacked learners have been around for quite some time.  This is an experiment in stacking in a way that is similar to standard deep learning, but without back propagation.  The approach uses functional link neural networks, where the non-linearity is in the synapse instead of the neuron (similar to a kernel method).  This allows a linear solve to by applied to each layer (or each model in each layer) and therefore does not require back propagation.  A kernel is then applied to the output of the linear solve, and the process repeated.  In the end it looks very similar to layer by layer training of a deep neural network, however a linear solve is applied to each model at each layer.

## Tools

The parallel python tool Ray is used and so the models can be created using multicore or cluster parallelism

## Status

This is a work in progress


