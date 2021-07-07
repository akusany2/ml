import load_data
import model
import train
import helpers
import test
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
else:
    print("using CPU")
print(device)

# Load data and classes
data = load_data.Load_data()
# print(data.get_mean_std(data.trainloader))

# Load model
net = model.Net()
net.to(device)

# Train model
train = train.Train(net, device)
train.train_network(data.trainloader)
train.save(helpers.PATH)

# Test model
# test = test.Test(net, data.testloader, data.classes)
# test.test_network_with_limit(4)
# test.test_network_full()
# test.performance()
