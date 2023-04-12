# SP23-CSE-60868-NeuralNetworks-FinalProject-InProgress
Paintings Denoising and Anomaly Detection Using Auto-Encoders (AEs)
## 1. Dataset
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
Dataset II: after training 20 epochs, training loss = 0.xx, valid loss = 0.xx, testing loss = 0.xx <br />
### (3) Model 3
TODO

## 4. Discussion
(1) Model 1 - AEs using feedfoward neural networks for image reconstruction, performs okay on the simple Dataset I for reconstruction while pretty bad on the more complicated DatasetII
