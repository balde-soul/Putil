import argparse

options = argparse.ArgumentParser()
args = options.parse_args()

from enum import Enum

class a(Enum):
    a = 0

print(a.a.name)
print(a.a in a)


t = (1, 2)
t = (1, 2)
a = 0