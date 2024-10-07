# Adversarial_Attacks
This project aims to evaluate the robustness of different models against different adversarial attack algorithms.
The current version has integrated 4 different neural network architectures that you can use to visualize the impact of various Adversarial attacks such as the FGSM Attack.

# Models Implemented
1. ResNet (results of robustness evaluation of the ResNet18 model specifically has been uploaded, however you can use the ResNet34, ResNet 50, and other variations as well)
2. MobileNet V2
3. DLA Model
4. VGG Model

# Future Work
Apart from the Fast Gradient Signed Method attack and the Project Gradient Descent Attack (PGD), future work can include implementing the Carlini & Wagner Attack (C&W), and the Basic Iterative Method Attack(BIM) algorithms on the same models and the dataset to check for the model robustness against different adversarial attack algorithms. The models can also further be trained with adversarial inputs in order to make them more robust against such attacks.

# Project Packages/Modules

**MODELS:** contain all of the different neural network architectures and models, the robustness of which is going to be tested against adversarial attacks.

**DATA:** contains the data loader along with data augmentation functionalities for the CIFAR-10 dataset.

**TRAINING:** contains the functionalities for training, calculating the accuracy, precision and recall of the different models.

**ADVERSARIAL_ALGORITHMS:** contains the implementation of different adversarial attacks such as the FGSM attack.

**UTILS:** contains functions for visualizing, saving and properly evaluating the robustness of different models against adversarial attacks.

**EXPERIMENTS:** contains model specific code for evaluating the model robustness of different architectures against adversarial attacks.

# Dataset Used
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset has been used for training the different models, and for applying the Adversarial attacks to check model robustness.

# RESULTS

| Model  | Accuracy | Accuracy after FGSM Attack | Accuracy after PGD Attack |
| ------------- | ------------- | ------------- | ------------- |
| [ResNet18](https://arxiv.org/abs/1512.03385) | 86.4%  | 29.69% | 17.57% |
| [VGG11](https://arxiv.org/abs/1409.1556) | 83.99% | 32.98% | 27.89% |
| [MobileNetV2](https://arxiv.org/abs/1801.04381) | 83.79% | 25.04% | 11.73% |
| [DLA](https://arxiv.org/pdf/1707.06484) | 84.35% | 21.13% | 11.38% |
