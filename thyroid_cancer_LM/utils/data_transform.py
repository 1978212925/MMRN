from torchvision.transforms import transforms

aug_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# imageNet的标准化[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
node_transform = {
    "train": transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]),
    "original": transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               ])}

transform_none = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor()])