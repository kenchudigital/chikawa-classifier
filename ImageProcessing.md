# Image Processing

## Basic Algorithms 

In this notebook, I will use algorithms to preprocess images in the image_labeler.ipynb notebook. Additionally, I will demonstrate how to deploy this preprocessing functionality in a graphical user interface (GUI).

You can create the dataset through the GUI by running the command:

`python image_labeler.py` Of course, we will not use this tiny dataset to train the model. It just provides the concept of the dataset.

The following is the basic Algorithms or knowledge of Image Processing:

- **Human Vision**: Understanding how humans perceive images, including color vision and visual perception mechanisms.
- **Luminance**: The measurement of brightness in images, often used in grayscale conversion and image analysis.
- **False Contouring**: Artifacts in images where color bands appear due to insufficient color depth or compression.
- **Saturation**: The intensity of color in an image, affecting how vivid or muted the colors appear.
- **Bit Plane Slicing**: Technique to separate an image into different bit planes to analyze or enhance specific aspects of the image.
- **Grey Level Transformation**: Techniques to adjust pixel intensity levels, including:
  - **Linear Transformation**: Adjusting pixel values using a linear function.
  - **Log Transformation**: Applying a logarithmic function to enhance low-intensity details.
  - **Gamma Correction**: Non-linear adjustment of pixel values to correct for display device characteristics.
  - **Thresholding**: Binarizing an image based on a specific intensity threshold.
- **Histogram Equalization**:
  - **Global**: Enhancing contrast by spreading out the intensity values over the entire range.
  - **Local**: Improving contrast in specific regions of the image using local histograms.
- **Filters**:
  - **Mean Filter**: Blurring and noise reduction by averaging pixel values in a neighborhood.
  - **Gaussian Filter**: Smoothing and noise reduction using a Gaussian function.
  - **Isotropic Filter**: Uniform smoothing in all directions.
  - **Adaptive Median Filter**: Noise reduction by adjusting the filter size based on local image characteristics.
- **FFT + Filter (Low-Pass) + Inverse FFT**: Fourier Transform to shift to the frequency domain, apply a low-pass filter, and then inverse Fourier Transform to get the filtered image.
- **Gradient (Edge Detection)**: Detecting edges by computing the gradient of pixel intensities.
- **Canny Edge Detection**: An edge detection algorithm that uses a multi-stage process to detect edges in images.
- **Harris Corner Detection**: Identifying corners in images using the Harris operator.
- **Camera Model & Stereo Vision**: Understanding how cameras capture images and using multiple cameras to perceive depth and 3D structure.
- **Camera Calibration**: The process of determining the intrinsic and extrinsic parameters of a camera to correct lens distortions and align images.

## CNNs (Convolutional Neural Networks)

- **Basic CNNs**: Fundamental architecture involving convolutional layers, pooling layers, and fully connected layers.
- **R-CNN (Region-based CNN)**: An extension of CNNs for object detection that involves region proposals and classification.
  - [R-CNN Paper](https://arxiv.org/abs/1311.2524)
- **AlexNet**: A deep CNN architecture that won the ImageNet competition in 2012, known for its depth and use of ReLU activation functions.
  - [AlexNet Paper](https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
- **VGGNet**: A deep CNN architecture with a simple and uniform structure using small 3x3 convolutional filters.
  - [VGGNet Paper](https://arxiv.org/abs/1409.1556)
- **GoogleNet**: Introduced the Inception module, which improves computational efficiency and performance by using multiple filter sizes in parallel.
  - [GoogleNet Paper](https://arxiv.org/abs/1409.4842)
- **ResNet**: Introduced residual connections to address the vanishing gradient problem, allowing for very deep networks.
  - [ResNet Paper](https://arxiv.org/abs/1512.03385)

## RNNs (Recurrent Neural Networks)

- **Basic RNNs**: Neural networks with connections that form directed cycles, allowing them to maintain a memory of previous inputs.
  - [Basic RNN Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **LSTM (Long Short-Term Memory)**: A type of RNN designed to overcome issues with long-term dependencies, using gates to manage information flow.
  - [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)

## GEN (Generative Models)

- **Auto-Encoder**: A type of neural network used to learn efficient representations of data, typically for dimensionality reduction or denoising.
  - [Auto-Encoder Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608011002059)
- **VAE (Variational Auto-Encoder)**: A generative model that learns to encode data into a latent space and decode it back, capturing data distribution.
  - [VAE Paper](https://arxiv.org/abs/1312.6114)
- **GAN (Generative Adversarial Network)**: A framework with two networks, a generator and a discriminator, that compete to improve the quality of generated samples.
  - **C-GAN (Conditional GAN)**: A GAN variant that generates images conditioned on additional input data.
    - [C-GAN Paper](https://arxiv.org/abs/1411.1784)
  - **DC-GAN (Deep Convolutional GAN)**: Uses deep convolutional networks for both generator and discriminator, improving image generation quality.
    - [DC-GAN Paper](https://arxiv.org/abs/1511.06434)
  - **C-DC-GAN (Conditional Deep Convolutional GAN)**: Combines conditional and deep convolutional GAN approaches for better control over generated images.
  - **AttentionGAN**: Enhances GANs with attention mechanisms to focus on relevant parts of the input data.
    - [AttentionGAN Paper](https://arxiv.org/abs/1805.08014)
  - **StyleGAN**: A GAN architecture known for generating high-quality images with fine control over style and features.
    - [StyleGAN Paper](https://arxiv.org/abs/1812.04948)
- **Diffusion Model**: A generative model that gradually denoises a sample to generate data.
  - **Fine-Tune**: Techniques to customize the diffusion model for specific tasks or datasets, such as DreamBooth, TextInversion, LoRa.
    - [DreamBooth Paper](https://arxiv.org/abs/2208.12242)
    - [TextInversion Paper](https://arxiv.org/abs/2306.08955)
    - [LoRa Paper](https://arxiv.org/abs/2003.07285)
  - **More Techniques**:
    - **Image2Image**: Converting one image to another image using a diffusion model.
    - **SD_depth**: Adding depth information to images using diffusion models.
    - **ControlNet**: A method to guide image generation by additional control signals.
    - **InPaint**: Filling in missing parts of images using diffusion models.
- **Autoagressive Models**: Models that generate data sequentially, where each step depends on previous steps, such as PixelCNN or PixelSNAIL.
  - [PixelCNN Paper](https://arxiv.org/abs/1606.05328)
  - [PixelSNAIL Paper](https://arxiv.org/abs/1707.03483)

## ViTs (Vision Transformers)

- **Transformer (Self-Attention)**: A model architecture that uses self-attention mechanisms to capture relationships between different parts of an image or sequence, providing a powerful alternative to CNNs for image processing.
  - [Attention is all you need](https://arxiv.org/abs/1706.03762) 
  - [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)

## GNNs (Graph Neural Networks)

- **Basic GNNs**: Models designed to work with graph-structured data, learning representations based on node features and graph topology.
  - [GNN Paper](https://arxiv.org/abs/1609.02907)

## NeRF (Neural Radiance Fields)

- **NeRF (Neural Radiance Fields)**: A technique for representing 3D scenes using neural networks. NeRF models scenes by learning a continuous volumetric representation, which can generate novel views of a scene from a set of input images.
  - [NeRF Paper](https://arxiv.org/abs/2003.08934)