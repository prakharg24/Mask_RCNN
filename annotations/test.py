import sys
import json

with open(sys.argv[1], 'r') as fp:
	data = json.load(fp)

for ele in data:
	print(ele)