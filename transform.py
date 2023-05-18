from torchvision import transforms


def train_transform() :
    return transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                              ])    

def val_transform() :
    return transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                              ])