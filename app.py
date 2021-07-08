import load_data
import model
import train
import helpers
import test
import torch, gc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

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
# test = test.Test(net, data.testloader, data.classes, device, data.batch_size)
# test.test_network_with_limit()  # number of classes = batch size
# test.test_network_full()
# test.performance()
