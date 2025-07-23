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