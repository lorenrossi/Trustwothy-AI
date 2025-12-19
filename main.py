
# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# pip install torch
# pip install packaging
# pip install adversarial-robustness-toolbox
# pip install timm
# pip install matplotlib

import torch
import torchvision
import inspect
import numpy as np

from art.attacks import evasion
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniLInfMethod
from art.estimators.classification import PyTorchClassifier
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# from torchvision.datasets import (
#     MNIST,
#     FashionMNIST,
#     EMNIST,
#     KMNIST,
#     QMNIST,
#     CIFAR10,
#     CIFAR100,
#     SVHN,
#     STL10,
#     ImageNet,
#     ImageFolder
# )
# MNIST – handwritten digits
# FashionMNIST – clothes instead of digits
# EMNIST – extended MNIST (letters)
# KMNIST – Japanese characters
# QMNIST – improved MNIST
# CIFAR10 / CIFAR100 – small color images
# SVHN – house numbers
# STL10 – larger CIFAR-like dataset
# ImageNet – large-scale dataset (manual download required)
# ImageFolder – load your own dataset from folders


# training loop
def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0
    train_counter = 0

    for x, y in loader:

        train_counter = train_counter + 1
        print('train counter: ', train_counter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# evaluation loop
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    eval_counter = 0

    with torch.no_grad():
        for x, y in loader:

            eval_counter = eval_counter + 1
            print('eval counter: ', eval_counter)

            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


if __name__ == '__main__':

    # Check in GPU is available (sullo zbook solo CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # x = torch.randn(3, 3).to(device)
    # print(x.device)

    print('EVASION ATTACKS IN ART:')
    attacks = [
        name
        for name, obj in inspect.getmembers(evasion)
        if inspect.isclass(obj)
    ]
    for attack in sorted(attacks):
        print(attack)

    # ------------------------------------------------------
    # Upload dataset and pre-processing
    from torchvision.datasets import MNIST

    mean_Imagenet = np.array([0.485, 0.456, 0.406])
    std_Imagenet = np.array([0.229, 0.224, 0.225])

    # Transformations needed since we are going to use a model pre-trained on Imagenet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # train dataset
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    # test dataset
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    num_classes = len(train_dataset.classes)
    print("Training images:", train_size)
    print("Test images:", test_size)
    print("Number of classes:", num_classes)

    val_size = train_size//6 * 1  # 10_000
    train_size = train_size - val_size  # 50000
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size]
    )
    print("Training images:", len(train_dataset))
    print("Val images:", len(val_dataset))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('Length train loader: ', len(train_loader))
    print('Length val loader: ', len(val_loader))
    print('Length test loader: ', len(test_loader))

    print('Dataset and pre-processing finished')

    # ------------------------------------------------------
    # Upload pre-trained model and fine-tuning

    from torchvision.models import resnet18
    model = models.resnet18(pretrained=True)
    # model = models.resnet18(weights="IMAGENET1K_V1")

    # Replacing the head of the model
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Freezing the backbone of the model
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    # Loss and optimizer for the tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=1e-3
    )

    print('-----------------------------------------------------------------')
    print('Started training')
    # Train and validate
    epochs = 1
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader)
        val_acc = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )
    print('Finished training')

    print('-----------------------------------------------------------------')
    print('Started testing')
    # Final test evaluation
    test_acc = evaluate(model, test_loader)
    print("Test accuracy:", test_acc)
    print('Finished testing')

    # Check on the model
    # print(model)
    # Conv2d(in_channels=3, out_channels=64, kernel_size=3, ...)
    # x = torch.randn(1, 3, 224, 224)  # batch_size = 1
    # out = model(x)
    # print(out.shape)  # [1, 10]

    # ------------------------------------------------------
    clip_min = ((0.0 - mean_Imagenet) / std_Imagenet).min()
    clip_max = ((1.0 - mean_Imagenet) / std_Imagenet).max()
    clip_values = (clip_min, clip_max)  # (0.0, 1.0)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=num_classes,
        clip_values=clip_values
    )

    # ------------------------------------------------------
    # Attacks implementation

    # test_images, test_labels = next(iter(test_loader))
    # test_images_np = test_images.numpy()
    # test_labels_np = test_labels.numpy()
    # test_labels_np = np.eye(10)[test_labels]  # one-hot encoding for ART
    # test_images_np, test_labels_np = test_images_np.to(device), test_labels_np.to(device)

    epsilon = 6 / 255

    attack = FastGradientMethod(classifier, eps=epsilon)
    # attack = ProjectedGradientDescent(
    #     estimator=classifier,
    #     norm=np.inf,  # L∞ PGD (most common)
    #     eps=0.3,  # max perturbation
    #     eps_step=0.01,  # step size
    #     max_iter=40,  # number of iterations
    #     targeted=False
    # )
    # attack = CarliniLInfMethod(
    #     classifier=classifier,
    #     confidence=0.0,
    #     targeted=False,
    #     learning_rate=0.01,
    #     max_iter=40,
    #     eps=0.3
    # )

    # Calcolo robust accuracy batch-wise
    num_correct = 0
    num_samples = 0
    counter_test_attacks = 0

    for test_images, test_labels in test_loader:

        # Single batch attack
        counter_test_attacks = counter_test_attacks + 1
        print(counter_test_attacks)

        # ART requires numpy
        test_images_np = test_images.numpy()
        test_labels_np = test_labels.numpy()

        # Genera adversarial batch
        # print('Started attack')
        adv_test_images = attack.generate(x=test_images_np)
        # print('Finished attack')

        # plt.subplot(1, 2, 1)
        # plt.title("Original")
        # plt.imshow(adv_test_images[0][0], cmap='gray')
        #
        # plt.subplot(1, 2, 2)
        # plt.title("Adversarial")
        # plt.imshow(adv_test_images[0][0], cmap='gray')
        # plt.show()

        # Prediction on the adversarial batch - ART automatically set no-grad inside the classifier
        pred_logits = classifier.predict(adv_test_images)

        acc = np.mean(np.argmax(pred_logits, axis=1) == test_labels_np)
        # print("Adversarial accuracy:", acc)

        if pred_logits.shape[1] > 1:  # multi-class
            pred_test_labels = np.argmax(pred_logits, axis=1)
        else:
            pred_test_labels = (pred_logits > 0.5).astype(int)

        num_correct += np.sum(pred_test_labels == test_labels_np)
        num_samples += test_images.shape[0]

    robust_accuracy = num_correct / num_samples
    print("Adversarial Test Accuracy:", robust_accuracy)

    attack_success_rate = 1 - robust_accuracy
    print("Test "
          ""
          "Attack Success Rate:", attack_success_rate)


