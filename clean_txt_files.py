import os

folder = 'tiny-imagenet-200'
for (path,dir,filenames) in os.walk(folder):
	for filename in filenames:
		if '.txt' in filename:
			print(filename)
			print(path)
			os.remove('/Users/Gustav/Documents/ADL/reimplementation_SCAN/AdvDL_project/'+path+'/'+filename)