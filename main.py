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
		words_5000.append([line[0]])
	f.close()
	
	
	f = open('input2.txt','r')
	origin = f.read()
	print('Origin paragraph:')
	print(origin,'\n')
	# paragraph is a list of origin paragraph split by ' '
	paragraph = origin.split(' ')

	#token_pos => [('Tokyo', 'NNP'), ('(', '('), ('CNN', 'NNP'), (')', ')'), ('Japan', 'NNP'), ("'s", 'POS'), ....]
	token_pos = nltk.pos_tag(nltk.word_tokenize(origin))
	print(token_pos)
	#word_token => dictionary with key = paragraph's index, value = word_tokenize(value of paragraph)
	#word_token => {0: ['Tokyo'], 1:['(', 'CNN', ')', 'Japan', "'s"], 2:['85-year-old'], ....}
	word_token = {}
	for index, token in enumerate(paragraph):
		word_token[index] = nltk.word_tokenize(token)
	
	#list of (word_token's index, pos, original token, origina token's lower and simple tense || original token)
	#allword => [(0, 'NNP', 'Tokyo', 'Tokyo'), (1, '(', '(', '('), (1, 'NNP', 'CNN', 'CNN'), .....]
	allword = []
	verb_pos = ['VBD','VBG','VBN','VBP','VBZ']
	noun_pos = ['NNS']
	posIndex = 0
	for key, value in word_token.items():
		for v in value:
			current_pos = token_pos[posIndex][1]
			#simplify the word tense of V. and N.
			if current_pos in verb_pos:
				word = (key, current_pos, v, WordNetLemmatizer().lemmatize(v.lower(),'v'))
				allword.append(word) 
			elif current_pos in noun_pos:
				word = (key, current_pos, v, WordNetLemmatizer().lemmatize(v.lower(),'n'))
				allword.append(word)
			else:
				word = (key, current_pos, v, v)
				allword.append(word)
			posIndex += 1
	

	# the POS we check whether the word is difficult.
	care_pos = ['NN', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',' VBZ']
	print('Difficult words:')
	for word in allword:
		if word[3] not in [i[0] for i in words_5000] and word[3].lower() not in [i[0] for i in words_5000] and word[1] in care_pos:
			print(word)
	