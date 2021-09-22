import torch
import torchvision.transforms.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class IMAGENET():
    def __init__(self, arch):
        self.model = models.__dict__[arch](
            pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to(device)

    def predict_label(self, image):
        image_copy = image.clone()
        image_copy = torch.stack([F.normalize(image_copy[i],
                                              [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
                                  for i in range(image.shape[0])])
        image = image_copy.to(device)
        with torch.no_grad():
            output = self.model(image)
        return torch.argmax(output).item()


    def predict_scores(self, image):
        image_copy = image.clone()
        image_copy = torch.stack([F.normalize(image_copy[i],
                                              [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
                                  for i in range(image.shape[0])])
        image = image_copy.to(device)
        output = self.model(image)
        return output

# load_imagenet_data
def MyDataset(size1=256, size2=224):
    dataset = datasets.ImageFolder(
        '../../../../../ILSVRC2012/val/',
        transform=transforms.Compose([
            transforms.Resize(size1),
            transforms.CenterCrop(size2),
            transforms.ToTensor(),
        ]))
    return dataset





