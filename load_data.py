import torch
import torchvision
import torchvision.transforms as transforms
import helpers


# 1. Load and normalize
class Load_data:
    batch_size = 1
    training_dataset_path = "./Dataset/train"
    testing_dataset_path = "./Dataset/val"

    def __init__(self):
        # mean & std = (tensor([1.0102, 0.9364, 0.8310]), tensor([0.6800, 0.6295, 0.5738]))
        mean = [1.0102, 0.9364, 0.8310]
        std = [0.6800, 0.6295, 0.5738]
        training_transform = transforms.Compose(
            [
                transforms.Resize((100, 100)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(torch.tensor(mean), torch.tensor(std)),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(torch.tensor(mean), torch.tensor(std)),
            ]
        )

        trainset = torchvision.datasets.ImageFolder(
            root=self.training_dataset_path, transform=training_transform
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        testset = torchvision.datasets.ImageFolder(
            root=self.testing_dataset_path, transform=test_transform
        )
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def get_mean_std(self, loader):
        mean = 0
        std = 0
        total_images_count = 0
        for images, _ in loader:
            image_count_in_a_batch = images.size(0)

            images = images.view(
                image_count_in_a_batch, images.size(1), -1
            )  # shape images
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images_count += image_count_in_a_batch

            mean /= total_images_count
            std /= total_images_count
            return mean, std

    classes = (
        "burger",
        "butter_naan",
        "chai",
        "chapati",
        "chole_bhature",
        "dal_makhani",
        "dhokla",
        "fried_rice",
        "idli",
        "jalebi",
        "kaathi_rolls",
        "kadai_paneer",
        "kulfi",
        "masala_dosa",
        "momos",
        "paani_puri",
        "pakode",
        "pav_bhaji",
        "pizza",
        "samosa",
    )
