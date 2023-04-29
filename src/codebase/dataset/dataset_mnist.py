import os

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset.utils_dataset import get_dataset_with_image_and_attributes


class Dataset_mnist(Dataset):
    def __init__(self, dataset, attributes, transform=None, show_image=False):
        self.dataset = dataset
        self.show_image = show_image
        self.transform = transform
        self.attributes = attributes

    def __getitem__(self, item):
        image = self.dataset[item][0]
        attributes = self.attributes[item]
        if self.transform:
            image = self.transform(image)
        return image, self.dataset[item][1], attributes,

    def __len__(self):
        return len(self.dataset)


class Dataset_mnist_for_explainer(Dataset):
    def __init__(self, dataset, dataset_path, file_name_concept, file_name_y, attribute_file_name, transform):
        self.transform = transform
        self.dataset = dataset
        print(os.path.join(dataset_path, file_name_concept))
        self.concepts = torch.load(os.path.join(dataset_path, file_name_concept))
        self.attributes = torch.load(os.path.join(dataset_path, attribute_file_name))
        self.y = torch.load(os.path.join(dataset_path, file_name_y))
        self.y_one_hot = one_hot(self.y.to(torch.long)).to(torch.float)

        print(self.concepts.size())
        print(self.attributes.size())
        print(self.y.size())
        print(len(self.y))

    def __getitem__(self, item):
        im = self.dataset[item][0]

        if im.getbands()[0] == 'L':
            im = im.convert('RGB')
        if self.transform:
            im = self.transform(im)
        return im, self.concepts[item], self.attributes[item], self.y[item], self.y_one_hot[item]

    def __len__(self):
        return self.y.size(0)


def test_dataset():
    print("MNIST")
    train_set, train_attributes = get_dataset_with_image_and_attributes(
        data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
        json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
        dataset_name="mnist",
        mode="train",
        attribute_file="attributes.npy"
    )

    # train_transform = get_transforms(size=224)
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    dataset_path = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mnist/t/lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE/dataset_g"
    dataset = Dataset_mnist_for_explainer(train_set, dataset_path, "train_proba_concepts.pt", "train_class_labels.pt",
                                          "train_attributes.pt", transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    data = iter(dataloader)
    d = next(data)
    print(d[0].size())
    print(d[1])
    print(d[2])
    print(d[3])
    print(d[4])

    # model = Classifier("Resnet_10", 1, False)
    # # model = Classifier("AlexNet", 1, False)
    # pred = model(d[0])
    # print(pred.size())
    # print(d[1].size())
    #
    # fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    # ax.imshow(np.transpose(d[0].numpy()[0], (1, 2, 0)), cmap='gray')
    # ax.text(10, 0, f"Label: {d[1][0].item()}", style='normal', size='xx-large')
    # plt.show()


if __name__ == '__main__':
    test_dataset()
