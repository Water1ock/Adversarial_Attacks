import argparse
from experiments.experiment_resnet import main as resnet_experiment

def run_resnet_experiment():
    # Hardcoded parameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 64
    epsilon = 0.1

    # Create a dummy argument list
    args = [
        '--num_epochs', str(num_epochs),
        '--learning_rate', str(learning_rate),
        '--batch_size', str(batch_size),
        '--epsilon', str(epsilon)
    ]

    # Parse the arguments to create a Namespace object
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epsilon', type=float, required=True)

    parsed_args = parser.parse_args(args)

    # Call the experiment function with the parsed arguments
    resnet_experiment(parsed_args)

if __name__ == "__main__":
    run_resnet_experiment()