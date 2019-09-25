# **MNIST softmax Classifier** 

## Submissions
`mnist.py` : Build a softmax Classifier of MNIST dataset. Get 91% accuracy on the test dataset using 1-layer MLP, and 94% accuracy using naive non-linear version network.  
`mnist_data` :  mnist data folder. I use pytorch dataset tool `torchvision.datasets.MNIST` and `torch.utils.data.DataLoader` to prepare  batch data of training and testing.

## 1. Describe of code's pipeline.
1. Download and prepare dataset using pytorch dataloader;
2. Give two version naive classifier: `Net_L` and `Net_NL`;
3. Using SGD optimizer with lr=0.01 and momentum=0.5 and train epoch=9;
4. Two finctions `train()` and `test()` give main data logic about;
5. I use pytorch's `log_softmax()` instead of `softmax()` for computing efficiecy of loss;
