import os
import nltk
from bs4 import BeautifulSoup 
import json 

ECTNestedDict = dict()
path = os.path.join(os.getcwd(), 'ECT/')

# create directory for saving text information from transcripts
out_json = 'ECTNestedDict'
out_json = os.path.join(os.getcwd(), out_json)
if not os.path.exists(out_json):
	os.mkdir(out_json)

# create directory for saving text information from transcripts
out_dir = 'ECTText'
out_dir = os.path.join(os.getcwd(), out_dir)
if not os.path.exists(out_dir):
	os.mkdir(out_dir)

dict_count = 0
for i, filename in enumerate(os.listdir(path)):

	currDict = dict()
	corpus = nltk.data.load(os.path.join(os.getcwd(), 'ECT/', filename), format='raw').decode('utf-8')

	outfile = open(os.path.join(out_dir, filename[:-4]  + 'txt'), 'w')

	soup = BeautifulSoup(corpus, 'html.parser')

	#Date
	currDict['Date'] = " ".join((soup.find('p', {"class":"p p1"}).get_text()).split(' ')[-7:])

	# Participants
	count = 0
	currDict['Participants'] = []
	for info in soup.find_all('p', {'class':'p p1'}):
		if info.string is not None:
			if info.find('strong') is None: 
				 currDict['Participants'].append(info.string)
			else:
				count += 1
		if count == 3:
			break

	# Presentation 
	captureText = False
	currDict['Presentation'] = dict()
	for info in soup.find_all('p'):
		if info.string is not None:
			if 'Question-and-Answer Session'.lower() in info.string.lower() and info.find('strong') is not None:
				# print(info)
				break 

			if info.find('strong') is not None and 'Participants' not in info.string:
				t_string = info.string
				# print(t_string)
				if t_string[-1] == ' ':
					t_string = t_string[:-1:]
				captureText = True
				presenter = t_string
				currDict['Presentation'][t_string] = ""
			elif captureText:
				# print(info.string)
				currDict['Presentation'][presenter] += info.string
				# print(currDict['Presentation']) 
	 
	# Questionnaire
	cap = False
	count = -1
	subcount = -1 
	currDict['Questionnaire'] = dict()
	for info in soup.find_all('p'):
		if info.string is not None: 
			if 'Question-and-Answer Session' in info.string and info.find('strong') is not None:
				cap = True
				continue
			if cap:
				if info.find('span', {'class':'question'}) is not None:
					# print(info)
					count += 1
					subcount = -1
					currDict['Questionnaire'][count] = dict()

				if info.find('strong') is not None:
					# print(info)

					if count == -1:
						count = 0
						currDict['Questionnaire'][count] = dict()

					subcount += 1
					t_string = info.string
					if t_string[-1] == ' ':
						t_string = t_string[:-1:]

					currDict['Questionnaire'][count][subcount] = dict()

					currDict['Questionnaire'][count][subcount]['Speaker'] = t_string
					currDict['Questionnaire'][count][subcount]['Remark'] = ""
				else :
					# print(info.string)
					if count == -1:
						continue 
					currDict['Questionnaire'][count][subcount]['Remark'] += info.string

	with open(os.path.join(out_json, filename[:-4] + 'json'), 'w+') as f:
		json.dump(currDict, f)

	# Extract text information from Dictionary and store into text file
	for key, value in currDict.items():
		outfile.write(f'{key} {value}\n')
	outfile.close()

	

