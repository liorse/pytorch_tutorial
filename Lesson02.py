# %%
#!pip install scikit-learn
from sklearn.datasets import make_circles

n_samples = 1000
X, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=42)
# %%
print(f"first 5 features: {X[:5]}")
print(f"first 5 labels {y[:5]}")
# %%
#!pip install pandas
import pandas as pd
circles = pd.DataFrame({"X1": X[:,0],
                        "X2": X[:,1],
                        "label": y})
circles.head(10)
circles.label.value_counts()

# %%
import matplotlib.pyplot as plt
plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu);

# %%
# turn the data into tensors and create a train and test split
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
#%%
X[:5], y[:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% test, 80% training
                                                    random_state=32)
len(X_train), len(X_test), len(y_test), len(y_train)
# %%
# training set
plt.scatter(x=X_train[:,0], y=X_train[:,1], c=y_train)
# test set
plt.scatter(x=X_test[:,0], y=X_test[:,1], c=y_test, cmap=plt.cm.RdYlBu)
# %%
# building a model
from torch import nn

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device is {device}")
# %%
# build the model 
class CircleModelV0(nn.Module):

    def __init__(self):
        super().__init__()
        # create 2 nn Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
    
# %%
# create an instance of model
model_0 = CircleModelV0().to(device)
model_0

# %%
# make predictions with the model
untrained_preds = model_0(X_test.to(device))
print(f"untrained_preds length {len(untrained_preds)} {untrained_preds.shape}")
print(f"first 10 predictions: {untrained_preds[:10]}")
# %%
# create a loss function
loss_fn = nn.BCEWithLogitsLoss()

# create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
# %%
# writing an evaluation metric
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    
    acc = (correct/len(y_pred)) * 100
    return acc

# %%
# test the forward pass
y_logit = model_0(X_test.to(device))[:5]

# use sigmoid on the logits
y_pred_prob = torch.sigmoid(y_logit)
y_pred_prob
y_pred_labels = torch.round(y_pred_prob)
y_pred_labels.squeeze()
# %%
# building the training loop
torch.manual_seed(42)

epoches = 100

# put the data on the device
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

for epoch in range(epoches):

    model_0.train()

    # 1. forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. calculate the loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. optimzer zero grad
    optimizer.zero_grad()

    # 4. Loss back propogation calculation
    loss.backward()

    # 5. step optimizer, correct according to new gardients
    optimizer.step()

    # 6. testing the new model weights
    model_0.eval()
    with torch.inference_mode():
        # 1. forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. caculate loss of test results
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | loss {loss:.3f} | accuracy {acc:.3f} | test loss {test_loss:.3f} | test accuracy {test_acc:.3f}")


# %%
