if __name__ == '__main__':
	f = open('input.txt','r')
	paragraph = []
	for line in f.readlines():
		line = line.strip('\n')
		paragraph = paragraph + line.split(' ')

	print(paragraph)
