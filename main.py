import sys
from experiments.experiment_resnet import main as resnet_experiment

def run_resnet_experiment():
    # Hardcoded parameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 64
    epsilon = 0.1

    # Create a dummy argument list
    sys.argv = [
        'main.py',  # Script name
        '--num_epochs', str(num_epochs),
        '--learning_rate', str(learning_rate),
        '--batch_size', str(batch_size),
        '--epsilon', str(epsilon)
    ]

    resnet_experiment()

if __name__ == "__main__":
    run_resnet_experiment()