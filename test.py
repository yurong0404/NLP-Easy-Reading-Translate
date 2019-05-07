import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')

if __name__ == '__main__':
	
	#read Oxford 5000 vocabulary from file
	f = open('5000_voc_Oxford.txt','r')
	words_5000 = []
	for line in f.readlines():
		line = line.lower()
		line = line.strip('\n')
		line = line.split('\t',1)
		words_5000.append([line[0],line[1]])
	f.close()
	
	'''
	#read input file and encode into token list "paragraph"
	f = open('input.txt','r')
	paragraph = []
	for line in f.readlines():
		#if line == '\n':
		#	continue
		line = line.lower()
		line = line.strip('\n')
		temp = []
		remove_token = [',','.','"',"'s","s'"]
		for i in line.split(' '):
			for rm_token in remove_token:
				print(rm_token,i)
				while rm_token in i:
					i = i.replace(rm_token, '')
			temp.append(i)
		paragraph = paragraph + temp
	f.close()
	print('Origin:\n')
	print(paragraph)
	verb_pos = ['VBD','VBG','VBN','VBP','VBZ']
	noun_pos = ['NNS']
	print('Difficult words\n')
	for index, word in enumerate(paragraph):
		if word == '':
			continue
		print(nltk.pos_tag([word]))
		if nltk.pos_tag([word])[0][1] in verb_pos:
			word = WordNetLemmatizer().lemmatize(word.lower(),'v')
		elif nltk.pos_tag([word])[0][1] in noun_pos:
			word = WordNetLemmatizer().lemmatize(word.lower(),'n')
		if word.lower() not in [i[0] for i in words_5000]:
			print(word)
	'''

	
	f = open('input.txt','r')
	origin = f.read()
	print('Origin paragraph:')
	print(origin,'\n')
	paragraph = nltk.word_tokenize(origin)
	#print(paragraph)
	token_pos = nltk.pos_tag(paragraph)
	#print(token_pos)
	verb_pos = ['VBD','VBG','VBN','VBP','VBZ']
	noun_pos = ['NNS']
	for index, word in enumerate(token_pos):
		if word[1] in verb_pos:
			paragraph[index] = WordNetLemmatizer().lemmatize(word[0].lower(),'v')
		elif word[1] in noun_pos:
			paragraph[index] = WordNetLemmatizer().lemmatize(word[0].lower(),'n')

	care_pos = ['JJ', 'NN', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBS', 'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',' VBZ']
	print('Difficult words:')
	for index, token in enumerate(paragraph):
		if token not in [i[0] for i in words_5000] and token.lower() not in [i[0] for i in words_5000] and token_pos[index][1] in care_pos:
			print(index, '\t', token, '\t', token_pos[index][1])