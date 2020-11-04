import os
os.chdir("../")

output = open("output/res.txt", "w")

def printToFile(out):##輸出至檔案
	print(out)
	print(out, file=output)