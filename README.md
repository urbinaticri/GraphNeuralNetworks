

## Graph Node Classification

Experiments of Graph Node Classification on [Cora Dataset](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid) (full).

*Experiment #1*:

Definition of a Graph Convolution Layer based on [Message Passing](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#MessagePassing).

The model structure is inspired on [SSP](https://paperswithcode.com/paper/optimization-of-graph-neural-networks-with) that obtain the SOTA in Node Classification on this dataset.

Achieved **86%** accuracy on test set.

An extract of the progress of how the model learn to embed classes (downscaled in 2D using TSNE algorithm):

![](./imgs/embeddings.gif)

