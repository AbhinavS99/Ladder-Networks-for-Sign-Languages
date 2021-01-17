# Ladder-Networks-for-Sign-Languages
<h2>Ladder Networks</h2>
Ladder networks are an amalgamation of 
supervised and unsupervised learning in deep neural networks. In ladder networks, both the supervised and the unsupervised components are trained simultaneously to minimize the sum of the supervised and unsupervised cost functions by backpropagation, avoiding the need of layer-wise pre-training. Ladder networks achieve state of the art performance in semi-supervised MNIST and CIFAR-10 classifications.

<br>
<div style="text-align:center"><img src="./images/ladder_net.png" /></div>.
Fig: 2 Layer Ladder Network

---
<h2>Project Description</h2>
<h3>Introduction</h3>
The project aims to demonstrate the ability of ladder networks to translate sign languages into text while using lesser data than the traditional models. Vast amount of labelled data in context sign languages is a rare occurence. Ladder networks are known to reduce the amount of labels required and thus can make the translation task feasible.
<h3>Dataset</h3>
Sign Language MNIST dataset from kaggle is used for this project. The dataset contains all the alphabets except J and Z because they are motion based and cannot be captured due to obvious reasons. In total there are 23 classes. The dataset contains 27455 training images 3586 validation images and 3586 test images. Each image is uni-channel i.e. grayscale image with dimensions as 28x28. The preprocessing stages include normalizing each pixel value from 0 to 1 to prevent any sort of overflows. No image augmentation was done whatsoever as the required amount of data was already sufficient. The dataset had a fair class balance wherein each class had an average of 1143 images. The maximum number of images for a class were 1294 and the least was 957.
<br><a href = "https://www.kaggle.com/datamunge/sign-language-mnist">Link to dataset</a>
<h3>Methodolgy</h3>
<h4>Ladder Network Overview</h4>
Ladder network is a semi-supervised learning network which combines a traditional Deep Neural Network and an autoencoder (used to reconstruct activation at each layer). The working of ladder network is explained as follows:
<ol>
  <li>input x is fed and output y is returned</li>
  <li>x is fed into the regular deep neural network. The values and the layers of this network are called "clean".</li>
  <li>x is fed into an auto-encoder. The auto-encoder has the same layers and the weights as that of the deep neural network. However Gaussian noise is added to each layer's input. The noise serves the purpose of regularization effect.</li>
  <li>The autoencoder then propogates backward and attempts to recreate each layers pre-activation (i.e. the dot products of weights of that layer and the input to that layer) with the help of a denoising function, the corrupted pre-activation value, and the previous layer's pre-activation.</li>
</ol>
<h3>References</h3>
<ol>
  <li>Rasmus, Antti, et al. “Semi-Supervised Learning with Ladder Networks.” Advances in Neural Information Processing Systems. 2015.</li>
  <li>Rasmus, Antti, Harri Valpola, and Tapani Raiko. “Lateral Connections in Denoising Autoencoders Support Supervised Learning.” arXiv preprint arXiv:1504.08215 (2015).</li>
  <li>Valpola, Harri. “From neural PCA to deep unsupervised learning.” arXiv preprint arXiv:1411.7783 (2014).</li>
</ol>
**Note:** Not an exhaustive list
