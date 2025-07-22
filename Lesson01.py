# %%
# program overview
what_were_covering = {1: "data (prepaer and load)",
                      2: "build a model",
                      3: "fitting the model to data (training)",
                      4: "making predictions and evaluating the model (inference)",
                      5: "saving and loading a model",
                      6: "putting it all together (end-to-end project)"}
# %%
import torch
from torch import nn # nn contains all the layers we need to build a model
import matplotlib.pyplot as plt

# %%
torch.__version__ # check the version of PyTorch
# %%
# create known parameters for a linear model
weight = 5.3
bias = 6.7

# create the data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
# X are features, and y are labels

# %%# visualize the data
plt.scatter(X, y)
# %%
print(len(X))
#create a train/test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# %%
def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    """
    Plots training data, test data and predictions.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
    plt.legend()
    plt.xlabel("Input feature")
    plt.ylabel("Target label")
    plt.title("Training and Testing Data with Predictions")
    plt.show()

# %%
plot_predictions(X_train, y_train, X_test, y_test)
# %%
class LinearRegressionModel(nn.Module):
    """
    A simple linear regression model.
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

# %%
# make a model from our class
torch.manual_seed(140)
model_0 = LinearRegressionModel()
list(model_0.parameters())
#model_0.state_dict()

# making predictions using torch.inferece_mode()
with torch.inference_mode():
    y_preds = model_0(X_test)

# print the predictions
#print(y_preds)

#plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)

#y_test - y_preds

# defining a loss function and an optimizer
loss_fn = nn.L1Loss() # Mean absolute errors
optimizer = torch.optim.SGD(params = model_0.parameters(), # our model parameters
                            lr=0.01) # learning rate


# creating a training loop
epochs = 10000
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):

    # put the model in training mode
    model_0.train()

    # calculate the forward pass
    y_pred = model_0(X_train)

    # calculate the loss 
    loss = loss_fn(y_pred, y_train)

    # zero the optimizer
    optimizer.zero_grad()

    # backpropogate on the loss
    loss.backward()

    # update the parameters after the backpropogation 
    optimizer.step()

    # testing
    # put the model in eval mode
    model_0.eval()

    with torch.inference_mode():

        # calclulate forward pass
        test_pred = model_0(X_test)

        # calculate the loss on the test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        # print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch}, MAE train loss {loss}, MAE test loss {test_loss}")

# %%
# plot the loss curve
plt.plot(epoch_count, train_loss_values, label = 'Train Loss')
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.title("Training and test loss curves")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend();

# %%
# making predictions using the model
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
y_preds

# %%
plot_predictions(predictions=y_preds)
# %%
