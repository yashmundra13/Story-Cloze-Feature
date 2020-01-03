import json
import random
import csv
from nltk.tokenize import wordpunct_tokenize

def fetch_data():
	tra = []
	val = []
	ts = []
	with open('train.csv') as training_f:
		training = csv.reader(training_f)
		next(training)
		for elt in training:
			doc = ''
			doc = elt[1] + ' ' + elt[2] + ' ' + elt[3] + ' ' + elt[4]
			d1 = doc + ' ' + elt[5]
			d2 = doc + ' ' + elt[6]
			if (elt[7] == '1'):
				tra.append((wordpunct_tokenize(d1),int(1)))
				tra.append((wordpunct_tokenize(d2),int(0)))
			else:
				tra.append((wordpunct_tokenize(d1),int(0)))
				tra.append((wordpunct_tokenize(d2),int(1)))
	with open('dev.csv') as valid_f:
		validation = csv.reader(valid_f)
		next(validation)		
		for elt in validation:
			doc = ''
			doc = elt[1] + ' ' + elt[2] + ' ' + elt[3] + ' ' + elt[4]
			d1 = doc + ' ' + elt[5]
			d2 = doc + ' ' + elt[6]
			if (elt[7] == '1'):
				val.append((wordpunct_tokenize(d1),int(1)))
				val.append((wordpunct_tokenize(d2),int(0)))
			else:
				val.append((wordpunct_tokenize(d1),int(0)))
				val.append((wordpunct_tokenize(d2),int(1)))
	with open('test.csv') as test_f:
		test = csv.reader(test_f)
		next(test)		
		for elt in test:
			doc = ''
			doc = elt[1] + ' ' + elt[2] + ' ' + elt[3] + ' ' + elt[4]
			d1 = doc + ' ' + elt[5]
			d2 = doc + ' ' + elt[6]
			ts.append((wordpunct_tokenize(d1),elt[0]))
			ts.append((wordpunct_tokenize(d2),elt[0]))

	return tra, val, ts
