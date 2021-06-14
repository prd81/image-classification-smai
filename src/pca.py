from os import listdir
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#parameters

inpath, outpath = 'dataset/', 'output/'

imlist = listdir(inpath)

size_new, size_old = (64, 64, 3), (256, 256)

n, d, k = len(imlist), size_new[0]*size_new[1]*3, 32

#collecting data from images

arr = zeros((n,d))

for j,i in enumerate(imlist):
    im = Image.open(inpath+i).resize(size_new[:2])
    arr[j,:] = reshape(asarray(im, order = 'F'), (d))

#evaluating principal components

z = arr - mean(arr,0)

eva,v = linalg.eig(dot(z,z.T))

seq = argsort(eva)[-1:-1-k:-1]

#taking running error component-wise

run = copy(arr)

errlis = []

basis = zeros((d, 3))

for i, j in enumerate(seq):
    col = dot(z.T,v[:,j])
    col /= linalg.norm(col)
    run -= outer(dot(arr,col),col.T)
    errlis += [linalg.norm(run)]
    if i<3:
        basis[:,i] = col

#get compressed images

t = arr - run

#save compressed images

for i in range(n):
    im_arr = reshape(uint8(t[i,:]), size_new)
    im = Image.fromarray(im_arr).resize(size_old)
    im.save(outpath+str(i+1)+'.jpg')


#linear plot

f1 = plt.figure(1, figsize = (8, 6))
plt.plot(list(range(k)),errlis,'')
plt.xlabel('no. of components')
plt.ylabel('mean square error')
plt.title('error vs no. of components')
f1.savefig('lin.png')

#semi-log plot

f2 = plt.figure(2, figsize = (8, 6))
plt.plot(list(range(k)),log(errlis),'')
plt.xlabel('no. of components')
plt.ylabel('log(mean square error)')
plt.title('log-error vs no. of components')
f2.savefig('log.png')

#scatterplots in 1-D, 2-D, 3-D

spread = dot(t, basis)

cx, cy, cz = [spread[:,i] for i in range(3)]

#1-D scatterplot

f3 = plt.figure(3, figsize = (8, 6))
plt.scatter(cx, zeros((n,1)), 3, 'red')
plt.title('1-D scatterplot')
plt.xlabel('val along 1st axis')
f3.savefig('1-D.png')


#2-D scatterplot

f4 = plt.figure(4, figsize = (8, 6))
plt.scatter(cx, cy, 3, 'red')
plt.title('2-D scatterplot')
plt.xlabel('val along 1st axis')
plt.ylabel('val along 2nd axis')
f4.savefig('2-D.png')


#3-D scatterplot

f5 = plt.figure(5, figsize = (8, 6))
axis = Axes3D(f5)
axis.scatter(cx, cy, cz, c='r', marker='o')
axis.set_title('3-D scatterplot')
axis.set_xlabel('val along 1st axis')
axis.set_ylabel('val along 2nd axis')
axis.set_zlabel('val along 3rd axis')
f5.savefig('3-D.png')

