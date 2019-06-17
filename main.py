import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import os
import requests
import json
from bs4 import BeautifulSoup
from collections import defaultdict
nltk.download('averaged_perceptron_tagger')

NGRAM_API_URI = "https://{0}.linggle.com/query/"
class Linggle:
    def __init__(self, ver='www'):
        self.ver = ver

    def __getitem__(self, query):
        return self.search(query)

    def search(self, query):
        query = query.replace('/', '@')
        req = requests.get(NGRAM_API_URI.format(self.ver) + query)
        results = req.json()
        return results.get('ngrams', [])
    
    
def extract(soup):
    word_list = []
    for term in soup.select('.pt-list-terms'):
        count = 0    
        for item in term.select('.pt-list-terms__item'):
            if item.select('.pt-list-rating__indicator--high'):
                for title in item.select('.pt-thesaurus-card__term-title'):
                    if title.select_one('.link--term'):
                        if count == 3:
                            break
                        else:
                            count += 1
                            word_list.append(title.select_one('.link--term').text)
    return word_list

def crawl(url):
    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }
    source_code = requests.get(url , headers=headers).content
    soup = BeautifulSoup(source_code, 'html.parser')
    return extract(soup)
    
def synonymsPT(word):
    r = 'https://www.powerthesaurus.org/'+ word +'/synonyms'
    synonymsList = crawl(r)
    max_count = 0
    max_word = word
    for synonym in synonymsList:
        count = sum([row[1] for row in ling[word+" _"]][:10]) + sum([row[1] for row in ling["_ "+word]][:10])
        s_count = sum([row[1] for row in ling[synonym+" _"]][:10]) + sum([row[1] for row in ling["_ "+synonym]][:10])
        if s_count > (2*count) and s_count > 800000:
            max_count = s_count
            max_word = synonym
    return max_word

def crawlCNN(url):
    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }
    source_code = requests.get(url , headers=headers).content
    soup = BeautifulSoup(source_code, 'html.parser')
    
    paragraph = ""
    for big_block in soup.select('.zn-body__paragraph'):
        paragraph += big_block.text + '\n'
    
    return paragraph

def modifiedParagraph(origin, diffWord):
    origin = origin.split(' ')
    for word in diffWord:
        origin[word[0]] = origin[word[0]].replace(word[2], word[4])
    modified = ' '.join(origin)
    simpleWordList = [x[4] for x in diffWord]
    simpleWord = ' '.join(simpleWordList)
    return modified, simpleWord

if __name__ == '__main__':
	
	model = KeyedVectors.load_word2vec_format(os.getcwd()+"/glove_300d_word2vec.txt")

	#read Oxford 5000 vocabulary from file
	f = open('5000_voc_Oxford.txt','r')
	words_5000 = []
	for line in f.readlines():
	    line = line.lower()
	    line = line.strip('\n')
	    line = line.split('\t',1)
	    words_5000.append([line[0]])
	f.close()

	f = open('input.txt','r')
	origin = f.read()

	CNN_URL = 'https://edition.cnn.com/2019/06/15/asia/hong-kong-extradition-law-china-intl-hnk/index.html'
	origin = crawlCNN(CNN_URL)

	print('Origin paragraph:')
	print(origin,'\n')
	# paragraph is a list of origin paragraph split by ' '
	paragraph = origin.split(' ')

	#token_pos => [('Tokyo', 'NNP'), ('(', '('), ('CNN', 'NNP'), (')', ')'), ('Japan', 'NNP'), ("'s", 'POS'), ....]
	token_pos = nltk.pos_tag(nltk.word_tokenize(origin))

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
	        if posIndex >= len(token_pos):
	            break
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

	ling = Linggle()
	#the POS we check whether the word is difficult.
	care_pos = ['NN', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',' VBZ']
	#print('Difficult words:')
	diffWord = []
	dont_care_word = ['``', ',', '.', "'d", "'s", "''"]
	for word in allword:
	    if word[2] in dont_care_word or word[3] in dont_care_word:
	        continue
	    if word[3] not in [i[0] for i in words_5000] and word[3].lower() not in [i[0] for i in words_5000] and word[1] in care_pos:
	        try:
	            count = sum([row[1] for row in ling[word[2]+" _"]][:10]) + sum([row[1] for row in ling["_ "+word[2]]][:10])
	        except:
	            continue
	        #如果字在linggle的次數大於150萬，就篩掉，不視為困難字
	        if count < 1500000:
	            #==================== 用word embedding換同義字 ==========================
	            #similar word => 用word embedding前十相近的字挑出分數大於0.6的
	            try:
	                similar_word = [row[0] for row in model.most_similar(word[2], topn=10) if row[1] > 0.6]
	                max_count = 0
	                max_word = word[2]
	                for s_word in similar_word:
	                    s_count = sum([row[1] for row in ling[s_word+" _"]][:10]) + sum([row[1] for row in ling["_ "+s_word]][:10])
	                    #從similar word挑出在linggle次數最多的，且次數必須大於原字的2倍，且次數要大於120萬
	                    if s_count > (2*count) and s_count > max_count and s_count > 1200000:
	                        max_count = s_count
	                        max_word = s_word
	                if word[2] != max_word:
	                    word = word+(max_word, 'word embedding')
	                    diffWord.append(word)
	                #=============== 如果沒利用word embedding換字，則使用power thesaurus網佔找同義字 =================
	                else:
	                    replace_word = synonymsPT(word[2])
	                    if replace_word == word[2] or replace_word == word[3]:
	                        continue
	                    else:
	                        word = word + (replace_word, 'PT website')
	                        diffWord.append(word)
	            except:
	                continue