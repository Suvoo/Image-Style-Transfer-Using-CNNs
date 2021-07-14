# Implementation of Image Style Transfer Using Convolutional Neural Networks using Pytorch

## Style Transfer with Deep Neural Networks


This notebook *recreates* a style transfer method that is outlined in the paper, [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in PyTorch.

In this paper, style transfer uses the features found in the 19-layer VGG Network, which is comprised of a series of convolutional and pooling layers, and a few fully-connected layers. In the image below, the convolutional layers are named by stack and their order in the stack. Conv1_1 is the first convolutional layer that an image is passed through, in the first stack. Conv2_1 is the first convolutional layer in the *second* stack. The deepest convolutional layer in the network is conv5_4.

<img src='notebook_ims/vgg19_convlayers.png' width=80% />

### Separating Style and Content

Style transfer relies on separating the content and style of an image. Given one content image and one style image, we aim to create a new, _target_ image which should contain our desired content and style components:

An example is shown below, where the content image is of a cat, and the style image is of [Hokusai's Great Wave](https://en.wikipedia.org/wiki/The_Great_Wave_off_Kanagawa). The generated target image still contains the cat but is stylized with the waves, blue and beige colors, and block print textures of the style image!

<img src='notebook_ims/style_tx_cat.png' width=80% />

This notebook will use a pre-trained **VGG19 Net** to extract content or style features from a passed in image.

![image](https://user-images.githubusercontent.com/52796258/125573551-241921a3-f3fa-41ff-8cb5-e1a284f9f211.png)


### Results from the Notebook

![image](https://user-images.githubusercontent.com/52796258/125572547-0fe979ee-43b4-4a66-b246-ad474d76c2ab.png)

![image](https://user-images.githubusercontent.com/52796258/125572600-10c59e9a-d6af-4107-ae06-694534152118.png)

### Results from the Paper

![image](https://user-images.githubusercontent.com/52796258/125572846-bb29c915-52f3-48dd-9d6d-e1540448de09.png)

<h2 align= "left"><b>Project Maintainer(s)</b></h2>

<table>
<tr align="center">
  
  <td>
  
Suvodeep Sinha

<p align="center">
<img src = "https://avatars1.githubusercontent.com/u/52796258"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/Suvoo"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/suvodeep-sinha-59652418b/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
