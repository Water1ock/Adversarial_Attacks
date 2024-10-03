# Adversarial_Attacks
This project aims to evaluate the robustness of different models against 3 adversarial attack algorithms.
The current version has integrated 4 different neural network architectures that you can use to visualize the impact of various Adversarial attacks such as the FGSM Attack, namely:

1. ResNet (results of robustness evaluation of the ResNet18 model specifically has been uploaded, however you can use the ResNet34, ResNet 50, and other variations as well)
2. MobileNet V2
3. DLA Model
4. VGG Model

Currently, only the Fast Gradient Signed Method attack has been implemented, however, other attacks such as the Project Gradient Descent Attack (PGD), Carlini & Wagner Attack (C&W), and the Basic Iterative Method Attack(BIM) are going to be included as well. 

The project packages are used for the following:-

MODELS: contain all of the different neural network architectures and models, the robustness of which is going to be tested against adversarial attacks.
DATA: contains the data loader along with data augmentation functionalities for the CIFAR-10 dataset.
TRAINING: contains the functionalities for training, calculating the accuracy, precision and recall of the different models.
ADVERSARIAL_ALGORITHMS: contains the implementation of different adversarial attacks such as the FGSM attack.
UTILS: contains functions for visualizing, saving and properly evaluating the robustness of different models against adversarial attacks.
EXPERIMENTS: contains model specific code for evaluating the model robustness of different architectures against adversarial attacks.
