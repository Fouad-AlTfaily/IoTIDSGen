from collections import OrderedDict
from typing import Dict, Tuple
import argparse
import logging
import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from model.model import SimpleFNN, test, train
from helpers.dataset_partition import load_dataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

parser.add_argument(
    "--server_address", type=str, default="server:8080", help="Address of the server"
)
parser.add_argument(
    "--client_id", type=int, default=1, help="Unique ID for the client")

parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="Learning rate for the optimizer"
)


args = parser.parse_args()

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.trainloader, self.testloader = load_dataset(self.args.client_id, self.args.batch_size)
        self.model = SimpleFNN() 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        epochs = 1
        optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters(config), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.testloader, self.device)

        return float(loss), len(self.testloader), {"accuracy": accuracy}

# Function to Start the Client
def start_fl_client():
    try:
        client = FlowerClient(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()