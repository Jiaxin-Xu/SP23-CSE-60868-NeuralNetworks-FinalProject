# SP23-CSE-60868-NeuralNetworks-FinalProject
Paintings Denoising and Anomaly Detection Using Auto-Encoders (AEs)
## 1. Dataset
NOTE: Please see pdf files of part1 and part2 for further details of datasets and data curation <br />
TODO --- File too large to upload!! <br />
I: The Fashion MNIST dataset <br />
II: The Edvard Munch Paintings dataset <br />
III: The Van Gogh Paintings dataset <br />
## 2. Neural Networks Architecture
### (1) Model 1 - AEs using feedfoward neural networks for image reconstruction
Encoder structure TODO: one hidden layer MLP with ReLU activation <br />
Decoder structrue TODO: one hidden layer MLP with Sigmoid activation <br />
Input dimension TODO = 28 * 28 (Dataset I) or 256 * 256 * 3 (Dataset II&III) <br />
Latent space dimension TODO = 64 (Dataset I) or XXX (Dataset II&III) <br />
Loss function: MSE Loss
Optimization algorithm: Adam, with lr=1e-3
### (2) Model 2 - AEs using convolutional neural networks for image denoising
    For Dataset I:
    Autoencoder_Denoise(
    (encoder): Sequential(
      (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (decoder): Sequential(
      (0): ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ConvTranspose2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (4): Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    )
    For Dataset II:
    Autoencoder_Denoise(
    (encoder): Sequential(
     (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
     (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (2): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
     (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (decoder): Sequential(
     (0): ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (2): ConvTranspose2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
     (3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (4): Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    )
Loss function: MSE Loss
Optimization algorithm: Adam, with lr=1e-3
### (3) Model 3 - AEs using convolutional neural networks for image anomaly detection
TODO
## 3. Model Performance
### (1) Model 1
Dataset I: after training 10 epochs, training loss = 0.011862, valid loss = 0.002952, testing loss = 0.011870 <br />
Dataset II: after training 20 epochs, training loss = 0.046850, valid loss = 0.005557, testing loss = 0.046179 <br />
### (2) Model 2
Dataset I: after training 10 epochs, training loss = 0.009652, valid loss = 0.002408, testing loss = 0.009630 <br />
Dataset II: after training 20 epochs, training loss = 0.006298, valid loss = 0.000735, testing loss = 0.006197 <br />
### (3) Model 3
TODO

## 4. Discussion
(1) Model 1 - AEs using feedfoward neural networks for image reconstruction. It performs okay on the simple Dataset I for reconstruction while pretty bad on the more complicated DatasetII, where only vague images can be reconstructed. This is probably because unsqueeze the 3-channel color image into a 1D vector destroys the input data structure, thus cannot be reconstructed using a feedfoward neural networks. And that's why we are moving to Model 2, CNN. <br />
(2) Model 2 - AEs using convolutional neural networks for image denoising. However, the denoising performance on the test data of Dataset I is not very ideal. The denoised Fashion image is very vague. The trained denoising model on Dataset I was further tested on an out-of-sample dataset the MNIST dataset. The testing loss on MNIST dataset is very close to that on Dataset I, testing loss = 0.011131. Similar denoising performance was achieved on MNIST dataset, showing a good generalizability of the denoising model. In the final submission, I will probably try more complicated CNN structures, and searching for the best parameters. The denoising performance on the test data of Dataset II looks good now, but trying more complicated CNN structures and searching for the better parameters are also needed. <br />
(3) Model performance (loss) on validation set is always way better than the training and testing, while that of training and testing are always very close to each other. <br />
(4) Another task for before the final submission is the anamoly detection using AEs, on based the loss function of model trained on Dataset II, to do a "classification" task on distinguishing Dataset II and Dataset III. <br />
