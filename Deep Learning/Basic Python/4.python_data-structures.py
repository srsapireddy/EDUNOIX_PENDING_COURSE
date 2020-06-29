# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:42:46 2020

@author: Rahul Sapireddy
"""

# tuple - ordered and only redeable data
a =(1,2,3)
print(a) 

# list - can be changed accordingly
mylist = [1,2,3]
print("Zeroth Value: %d" % mylist[0])
mylist.append(4)
print("List length: %d" % len(mylist))
for value in mylist:
    print(value)
    
# dictionaries - have name or labels for the values
mydict = {'a':1,'b':2,'c':3}
print("A value: %d" %mydict['a'])
mydict['a'] = 11
print("A value: %d" %mydict['a'])
print("keys: %s" % mydict.keys())
print("Values: %s" % mydict.values())
for key in mydict.keys():
    print(mydict[key])
