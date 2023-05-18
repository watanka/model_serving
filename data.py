import torchvision
## adversarial attack scenario 가정

# separate, because we might need later for the deployment situation
def load_mnist(root, train, transform = None) :
    return torchvision.datasets.MNIST(root, train, transform = transform, download = True)