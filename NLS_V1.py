import re
from nltk.corpus import stopwords
import numpy as np
import os

essaydir = '/home/dmee/Practice/Dblock/'
scorebank = '/home/dmee/Practice/Scorebank.txt'

#After creating a bank and putting that in a text document, I needed to be able to read it back in for comparison. The code below does this. 

file = open(scorebank)
rawbank = file.read()
file.close()
bank = re.split("\)", rawbank)
wordbank = [(int, list)]*123
for (i,line) in enumerate (bank):
    if "5" in line[:5]:
        score = 5
    if "4" in line[:5]:
        score = 4
    if "3" in line[:5]:
        score = 3
    words = re.findall("(?<=\')[a-z]+(?=\')", line)
    wordbank[i] = (score, words)
    
#The below version is meant to draw from a folder of files, for example 15 student essays. Each of these essays are fairly uniform since students are submitting them in MLA format. This makes the cleaning process a bit easier. 

files = os.listdir(essaydir)
files = sorted(files)
allessays = []
for file in files:
    file = open(essaydir + file)
    rawtext = file.read()
    file.close()
    rawtext = rawtext.splitlines()
    text = [line for line in rawtext if len(line) > 6] #Get rid of some junk lines
    text = [re.sub(" {2,10}", "", line) for line in text] #I thought it was a tab at first, but it just formatted as multiple spaces
    text[1] = re.sub("[a-z]", "", text[1]) #
    allessays.append(text)
    
    
#The code below is the majority of the program. I decided to do it as a class in part as a way to practice, and in part because it seems like a tidy way to approach the process. I could see extracting some of the processes and experimenting with different python processes. For example, version 1 is simply interested in vocabulary similarity. I would like to create a version which coordinates this vocabulary similarity with a sentence structure similarity. The hypothesis being that not only do higher level students have a more robust vocabulary, but also a greater diversity of syntactic structures. 


class Essay:
    def __init__(self, essay):
        self.essay = essay
        self.metadata = essay[0:5]
        self.body = essay[5:] 
        self.quotedata = []#how many, average length, percentage of whole text
        self._tokenize()
        self._createlex()
        self._sentencescore()
    
    def _listoltolist(self, listol):
        singlelist = []
        for lines in listol:
            for l in lines:
                singlelist.append(l)
        return singlelist
    
    def _cleaner(self, dirtylist):
        cleanlist = []
        for line in dirtylist:
            line = line.lower()
            line = re.sub("\.{2,4}", "", line)
            line = re.sub("^ *$", "", line)
            if line != '':
                cleanlist.append(line)
        return cleanlist
    
    def _stopwords(self, tokens):
        sw = stopwords.words('english')
        swsupplemental = ['', 'hes', 'shes']
        for word in sw:
            if '\'' in word:
                word = re.sub("\'", "", word)
                swsupplemental.append(word)
        words = []
        for tok in tokens:
            if tok not in sw and tok not in swsupplemental:
                words.append(tok)
        return words
    
    def _cosim(self, d1, d2): #d1 essayscore, d2 scorebank
        num = len(set(d1).intersection(d2))
        d1len = np.sqrt(len(d1))
        d2len = np.sqrt(len(d2))
        denom = d1len * d2len
        if denom == 0: return 0
        return float(num)/float(denom)
    
    def _tokenize(self):
        #Preprocessing
        senlen = 0
        swsenlen = 0
        self.body = [re.sub("[^A-Za-z0-9“”.?! $\(\)]", "", line) for line in self.body] #scrub everything but quotes and punc 
        self.quotes = [re.findall("“[a-zA-Z0-9 $,.?’-]+”|\([0-9]+\)", line) for line in self.body] # Collects the quotes
        self.body = [re.sub("“[a-zA-Z0-9 $,.?’-]+”|\([0-9]+\)", "", line) for line in self.body] # Scrub the quotes
        
        #Processing
        self.body = self._cleaner(self.body)
        
        #Tokenizing
        self.sentences = []
        self.lex = []
        self.fulllex = []
        for (i, line) in enumerate (self.body):
            splice = re.split("[\?\.\!]", line) 
            for sen in splice:
                if len(sen) >= 3:
                    self.sentences.append((i,sen))

        for (i, sen) in enumerate (self.sentences):
            toksen = re.split(" ", sen[1]) #Here is the actual tokenization.
            for tok in toksen: #Man those pesky '' just want to show up everywhere!
                if tok == '':
                    toksen.remove('')
            senlen += len(toksen)
            sentotal = i
            self.fulllex.append(toksen)
            toksen = self._stopwords(toksen) #Get rid of the stop words
            swsenlen += len(toksen)
            self.lex.append(toksen)
            self.sentences[i] = (sen[0], toksen) #Capture the paragraph sources, as well as the tokenized sentence. 
            #  to replace the set number above.
        self.senlenavg = senlen/sentotal
        self.swsenlenavg = swsenlen/sentotal
        
    def _createlex(self):
        self.lex = self._listoltolist(self.lex)
        self.fulllex = self._listoltolist(self.fulllex)
        self.toklex = {}
        for word in self.fulllex:
            if word in self.toklex:
                self.toklex[word] += 1
            else:
                self.toklex[word] = 1
        self.typelex = set(self.lex)
        #a set of every word in the essay minus the stop words 
        #a dict with every word and a count in the essay
        self.toklexlist = list(self.toklex.items())
        self.toklexlist.sort(key=lambda x:x[1], reverse = True,)
    
    def _sentencescore(self):
        
        self.simscoresall = []
        
        for sen in self.sentences:
            self.simscoresword = []
            y = 0
            for bank in wordbank:
                coval = self._cosim(sen[1], bank[1])
                package = (coval, bank[0], bank[1])
                self.simscoresword.append(package)
            self.simscoresword = sorted(self.simscoresword, reverse = True)
            for cov in self.simscoresword:
                if cov[0] < .1:
                    if y == 0:
                        y = 1
                        #This also means that I found a sentence that has no (or at least very trivial) similarity to any other sentence. 
                    break
                y += 1
                if y == 10: #cap it out at 10
                    break
            average = sum(x[1] for x in self.simscoresword[:y])/y
            self.simscoresall.append((average, sen[1], self.simscoresword[:y])) #Easy to put the comparative sentences here, just removed for export.
        
        superavg = [int(0), int(0)]
        for scoreavg in self.simscoresall:
            superavg[0] += scoreavg[0]
            superavg[1] += 1
        self.avgfinal = superavg[0]/superavg[1]
        
#This is the end of the class, now time to create an instance, or in this case an instance for each text file in the accessed folder. 

allreports = []
for essay in allessays:
    report = Essay(essay)
    allreports.append(report)
    
#And now it's just time to test it out. Before I have it add to the text files, let's just have it print a few things out. 

for report in allreports:
	print(report.metadata)

#Now it's time to append the report on to the original document (i.e. the student's essay). It is worth noting that I was able to show this naive version to my own students. It was a fun lesson to talk about what the program captured well and what it didn't. I think the biggest surprise was the range of sentence length. What kind of correlation between student language skills and sentence length might be present? More investigation is merited. 


x = -1
for file in files:
    x += 1
    file = open(essaydir + file, "a")
    file.write("\nLanguage Analysis 1.0\n")
    file.write("Number of sentences\n")
    file.write(str((len(allreports[x].sentences))))
    file.write("\nNumber of Quotes\n")
    file.write(str((len(allreports[x].quotes))))
    file.write("\nAverage Length of Sentences\n")
    file.write(str((allreports[x].senlenavg)))
    file.write("\nAverage Length of Sentences without Stop Words\n")
    file.write(str((allreports[x].swsenlenavg)))
    file.write("\nNumber of different words\n")
    file.write(str((len(allreports[x].typelex))))
    file.write("\nTwenty Most common words, Including Stop Words\n")
    file.write(str(allreports[x].toklexlist[:20]))
    file.write("\nIndividual Sentences with Comparison Scores\n")
    for sen in allreports[x].simscoresall:
        file.write(str(sen))
        file.write(str("\n"))
    file.write("\nAverage Comparison Language Score\n")
    file.write(str(allreports[x].avgfinal))
    file.close()
    
#This concludes Version1 of the program. It is naive in the sense that it is only interested in comparing sentences with regard to vocabulary. This system could of course easily be gamed by a clever student who flooded their essay with awesome words, but of course at the current level this scorer is only anticipated to be used in conjunction with a human who is evaluating the other common criteria for essay assessment: Knowledge and understanding of the work studied, Appreciation of the Author's choices and style, and organizational structure. 

