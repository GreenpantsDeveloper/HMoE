# Hierarchical Mixture of Experts

This repository contains a **Hierarchical Mixture of Experts** (**HMoE**) deep neural network. This implementation allows for an **arbitrary depth** of the hierarchy. The Experts are set up as multilayer perceptrons which appear only at the leaf nodes of the hierarchy. All following layers contain a gating Manager that learns how much each of the Experts should contribute to the next layer in the hierarchy. The hierarchy is implemented as a **binary tree**, meaning that each Manager manages exactly two Experts.

In `hmoe.py` you can see how the network architecture is set up using the parameters found in `config.py`. To train the HMoE on MNIST, run:
```
python run_mnist_trainer.py
```

After 20 epochs (just over a minute of training on a GTX 1060 6GB), this architecture yields roughly **98% accuracy** on the test portion of MNIST (note that the test portion of MNIST is used as validation data here).
