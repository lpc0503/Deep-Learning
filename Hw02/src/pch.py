import os
os.chdir("../")

output = None

def openFile(filename):
	global output
	output = open("output/" + filename, "w")

def printToFile(out):##輸出至檔案
	print(out)
	print(out, file=output)