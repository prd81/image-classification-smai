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

v = dot(z.T,v[:,argsort(eva)[-1:-1-k:-1]]) #k principal eigenvectors

#processing labels

t = hstack((dot(x,v), ones((n,1)))) #transformed biased data matrix

label_index, index_label = {}, {}

k, m = min(k, n) + 1, 0 #no. of features, no. of classes

for label in label_list:
    if label not in label_index:
        label_index[label] = m
        index_label[m] = label
        m += 1

y = zeros((n,m))

for i, label in enumerate(label_list):
    y[i,label_index[label]] = 1

# gradient method using softmax function

w = zeros((k,m))

iter_max, eta = 10000, 1

for i in range(iter_max):
    h = dot(t, w)
    smax = exp(h - h.max(axis = 1, keepdims = True))
    smax /= sum(smax, axis = 1, keepdims = True)
    for j in range(m):
        w[:,j] += (eta/n)*dot(t.T, y[:,j]-smax[:,j])

#extract testing paths

test_list = []

test_file = argv[2]

with open(test_file,"r") as f:
    for line in f.readlines():
        path = line.strip("\n")
        test_list.append(path)

#test on testing data

out_list = []

for path in test_list:
    im = Image.open(path).convert('L').resize(size_new)
    test_vec = reshape(asarray(im),(d))
    trans = concatenate((dot(v.T,test_vec), [1]))
    cls_vec = dot(w.T, trans)
    cls = index_label[argmax(cls_vec)]
    out_list.append(cls)

#print labels

for cls in out_list:
    print(cls)

