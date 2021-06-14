from sys import argv
from numpy import *
from PIL import Image

#read training paths and labels

path_list, label_list = [], []

train_file = argv[1]

with open(train_file,"r") as f:
    for line in f.readlines():
        path, label = line.strip("\n").split()
        path_list.append(path)
        label_list.append(label)

#process data matrix and extract principal components

size_new = (64, 64)

n, d, k = len(path_list), size_new[0]*size_new[1], 32

x = zeros((n,d))

for i, path in enumerate(path_list):
    im = Image.open(path).convert('L').resize(size_new)
    x[i,:] = reshape(asarray(im),(d))

z = x - mean(x, 0)

eva, v = linalg.eig(dot(z,z.T))

v = dot(z.T,v[:,argsort(eva)[-1:-1-k:-1]]).T #k principal eigenvectors

#prepare training parameters

train_dict = {}

for i, label in enumerate(label_list):
    trans = dot(v,x[i,:])
    if label in train_dict:
        train_dict[label].append(trans)
    else:
        train_dict[label] = [trans]


mu, sigma, count = {}, {}, {}

for label in train_dict:
    arr = array(train_dict[label])
    train_dict[label] = arr
    mu[label] = mean(arr, 0)
    sigma[label] = var(arr, 0)
    count[label] = arr.shape[0]

#extract testing paths

test_list = []

test_file = argv[2]

with open(test_file,"r") as f:
    for line in f.readlines():
        path = line.strip("\n")
        test_list.append(path)

#test on testing data

out_list = []

bal, cf = 10**60, 1.33

for path in test_list:
    im = Image.open(path).convert('L').resize(size_new)
    test_vec = reshape(asarray(im),(d))
    trans = dot(v,test_vec)
    prob, cls = -float("inf"), None
    for label in train_dict:
        vec = (trans - mu[label])/sigma[label]
        probx = bal*(log(count[label]) - sum(log(sigma[label]))) - 0.5*linalg.norm((bal**cf)*vec)**2
        probx /= bal
        if probx > prob:
            prob, cls = probx, label
    out_list.append(cls)


#print labels

for cls in out_list:
    print(cls)

