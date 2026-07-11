import torch
from torch.utils.data import Dataset

REAL_REGISTRY = {"mnist", "fashionmnist"}

class RealImageDataset(Dataset):
    def __init__(self, name: str = "mnist", root: str = "./data", train: bool = True, subset_size: int = 8000, image_size: int = 28, seed: int = 0):
        """name = mnist or fashion"""
        import torchvision
        import torchvision.transforms as T

        name = name.lower()
        if name not in REAL_REGISTRY:
            raise ValueError(f"Unknown real dataset '{name}'. Choices: {sorted(REAL_REGISTRY)}")

        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

        if name == "mnist":
            ds = torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform)
        elif name == "fashion":
            ds = torchvision.datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
        else:
            raise NotImplementedError(f"Only MNIST and FashionMNIST are implemented, not {name}")

        self.channels = 1
        self.image_size = image_size
        self.name = name

        g = torch.Generator().manual_seed(seed)
        if subset_size is not None and subset_size < len(ds):
            idx = torch.randperm(len(ds), generator=g)[:subset_size].tolist()
        else:
            idx = range(len(ds))

        self.images = torch.stack([ds[i][0] for i in idx])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]