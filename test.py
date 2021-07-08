import torch
import torchvision
import helpers


class Test:
    def __init__(self, model, testloader, classes, device, batch_size):
        self.net = model
        self.device = device
        self.batch_size = batch_size

        # Load training model
        self.net.load_state_dict(torch.load(helpers.PATH))

        self.testloader = testloader
        self.classes = classes

    def test_network_with_limit(self):

        # 5. Test the network on the test data
        dataiter = iter(self.testloader)
        images, labels = dataiter.next()
        images, labels = images.to(self.device), labels.to(self.device)
        # print images
        helpers.imshow(torchvision.utils.make_grid(images))

        print(
            "GroundTruth: ",
            " ".join("%5s" % self.classes[labels[j]] for j in range(self.batch_size)),
        )

        outputs = self.net(images)

        _, predicted = torch.max(outputs, 1)

        print(
            "Predicted: ",
            " ".join(
                "%5s" % self.classes[predicted[j]] for j in range(self.batch_size)
            ),
        )

    def test_network_full(self):
        correct = 0
        total = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.net(images)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the 10000 test images: %d %%"
            % (100 * correct / total)
        )

    def performance(self):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        # again no gradients needed
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
