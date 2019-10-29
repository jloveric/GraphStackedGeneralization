# Graph Stacked Gneralization
Deep learning without back propagation.  Stacked generalization has been around for quite some time.  This is an experiment in stacking in a way that is similar to standard deep learning, but without back propagation.  In principle you can use any scikit learn ml algorithm ( random forest in many examples here) as your base model in the stacking network.  In each layer, additional models are produced based on the examples that fail the last trained model in that layer (this is different than gradient boosting).  In this way we produce something functionally similar to a set of eigenspaces or eigen models in each layer.  The class probabilities for each of those models are used as inputs to the next layer.  The "Graph" comes from the fact that one can struture the input with knowledge of the type of problem being solved.  Several examples that are very similar to "convolutional" layers are provided.  Only in the case of linear (or kernel methods) are the convolutional layers the same as those used in neural networks.  In the case of random forest, they are not in fact convolutions, they are just models over locally receptive fields.  I"ll provide a detailed writeup about what is going on here at some point.

## Tools

The parallel python tool Ray is used and so the models can be created using multicore or cluster parallelism.

## Status

Research project, this is huge work in progress and is far from ready for production.


