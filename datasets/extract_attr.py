# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
f = open('list_attr_celeba.txt')
newtxt = 'male.txt'
newnotxt = 'female.txt'
newf = open(newtxt, 'a+')
newnof = open(newnotxt, 'a+')

line = f.readline()
line = f.readline()
line = f.readline()
num_male = 0
num_female = 0
while line:
    array = line.split()
    if array[0] == "000001.jpg":
        print(array)
        print(array[21])
    if array[21] == '-1':
        new_context = array[0] + '\n'
        newnof.write(new_context)
        num_female += 1
    else:
        new_context = array[0] + '\n'
        newf.write(new_context)
        num_male += 1
    line = f.readline()

print('there are %d lines in %s' % (num_male, newtxt))
print('there are %d lines in %s' % (num_female, newnotxt))

f.close()
newf.close()
newnof.close()
