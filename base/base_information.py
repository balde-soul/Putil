# coding = utf-8
import os
import datetime
import sys

Date = datetime.datetime.now()
environ = os.environ
UserName = environ['USER']
UserHome = environ['HOME']
ExcutableName = os.path.split(sys.argv[0])[-1].split('.')[0]