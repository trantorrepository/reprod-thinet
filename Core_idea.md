## Pruning Convolutional Neural Networks for Resource Efficient Inference

[Paper Link](https://arxiv.org/abs/1611.06440)

<https://jacobgil.github.io/deeplearning/pruning-deep-learning>


First they state the pruning problem as a combinatorial optimization problem: choose a subset of weights B, such that when pruning them the network cost change will be minimal.

![](http://okye062gb.bkt.clouddn.com/2017-08-06-045633.jpg)

> s.t. stands for `subject to`, and the right equation means the number of zero should be less than B. `norm` means 基准

Notice how they used the absolute difference and not just the difference. Using the absolute difference enforces that the pruned network won’t decrease the network performance too much, but it also shouldn’t increase it. In the paper they show this gives better results, presumably because it’s more stable.

Now all ranking methods can be judged by this cost function.


### Oracle pruning

VGG16 has 4224 convolutional filters. **The “ideal” ranking method would be brute force - prune each filter, and then observe how the cost function changes when running on the training set.** Since they are from Nvidia and they have access to a gazillion GPUs they did just that. This is called the oracle ranking - the best possible ranking for minimizing the network cost change. Now to measure the effectiveness of other ranking methods, they compute the spearman correlation with the oracle. Surprise surprise(意外的惊喜), the ranking method they came up with (described next) correlates most with the oracle.

They come up with a new neuron ranking method based on a first order (meaning fast to compute) Taylor expansion of the network cost function.

Pruning a filter h is the same as zeroing it out.

C(W, D) is the average network cost function on the dataset D, when the network weights are set to W. Now we can evaluate C(W, D) as an expansion around C(W, D, h = 0). They should be pretty close, since removing a single filter shouldn’t affect the cost too much.

The ranking of h is then abs(C(W, D, h = 0) - C(W, D)).

![](http://okye062gb.bkt.clouddn.com/2017-08-06-052939.jpg)

![](http://okye062gb.bkt.clouddn.com/2017-08-06-053007.jpg)

The rankings of each layer are then normalized by the L2 norm of the ranks in that layer. I guess this kind of empiric(经验主义), and i’m not sure why is this needed, but it greatly effects the quality of the pruning.

This rank is quite intuitive. We could’ve used both the activation, and the gradient, as ranking methods by themselves. If any of them are high, that means they are significant to the output. Multiplying them gives us a way to throw/keep the filter if either the gradients or the activations are very low or high.

This makes me wonder - did they pose the pruning problem as minimizing the difference of the network costs, and then come up with the taylor expansion method, or was it other way around, and the difference of network costs oracle was a way to back up their new method ? :-)

In the paper their method outperformed other methods in accuracy, too, so it looks like the oracle is a good indicator(指示器).

Anyway I think this is a nice method that’s more friendly to code and test, than say, a particle filter, so we will explore this further!


### Pruning a Cats vs Dogs classifier using the Taylor criteria ranking

So lets say we have a transfer learning task where we need to create a classifier from a relatively small dataset.

Can we use a powerful pre-trained network like VGG for transfer learning, and then prune the network?

If many features learned in VGG16 are about cars, peoples and houses - how much do they contribute to a simple dog/cat classifier ?

This is a kind of a problem that I think is very common.

As a training set we will use 1000 images of cats, and 1000 images of dogs, from the Kaggle Dogs vs Cats data set. As a testing set we will use 400 images of cats, and 400 images of dogs.


#### Step one - train a large network

We will take VGG16, drop the fully connected layers, and add three new fully connected layers. We will freeze the convolutional layers, and retrain only the new fully connected layers. In PyTorch, the new layers look like this:

```
self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 2))
```

After training for 20 epoches with data augmentation, we get an accuracy of 98.7% on the testing set.
















