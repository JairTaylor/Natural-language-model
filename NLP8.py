# -*- coding: utf-8 -*-
#start on word-based model.

"""
Created on Wed Jan 13 20:56:39 2016

@author: Owner
"""
from __future__ import print_function
import numpy as np
import codecs
import sys
#import ngramtries4
#import temp

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:16:18 2016

@author: Owner
"""



from io import open
import numpy as np
import pickle #note: switch to Cpickle possibly in Python 2.x



#file = open('sentences.csv', encoding="utf8")


#total number of lines in senteces.csv: 4552600 
#number of english lines: 576612
# 

def sample(A, problookup = None):
    #given a list A of w's, sample an entry w[0] with probability proportional to w[1].
    #or if dictionary, sample a key with probability proportional to A[key].
    if type(A) is dict:
        keys = list(A.keys())
        if problookup is None:
            b = 0
            problookup = []
            for key in keys:
                b += A[key]
                problookup.append(b)  
        X = np.random.rand() * problookup[-1]        
        i = min([i for i in range(len(problookup)) if problookup[i] > X    ])  
        #print(i)
        return keys[i]

    if problookup is None:
        b = 0
        problookup = []
        for i in A:
            b += A[-1][1]
            problookup.append(b)        
    X = np.random.rand() * problookup[-1]
    i = min([i for i in range(len(A)) if problookup[i] > X    ])    
    return A[i][0]

class ngramtries():
    def __init__(self,numn = 2):
        self.numn = numn
        self.freq = [0,{}]
        
        
        #freq is a big tree giving the known n-grams and their frequencies.
        
        #freq is a list with two elements: freq[0] is the total number of characters seen,
        #freq[1] is a dictionary to all the subtrees.
        
        #each tree freq[1][x] for a character x is similarly a list T = [m, {a: , b: , c:...}] with two elements, where 
        #m is the number of times the character x has been seen; 
        #and T[1][a] is is a list [n, {...}] where n is the number of times xa has been seen. etc.s
        
        
        
        

    def clean(self,minfreq):
        #this code removes all ngrams appearing with frequency < minfreq.
        
        for g in self.allgrams(m = minfreq, make = "both", cleanup = True):
            pass
        #    print (g[0], g[1][0])
            
    def numgrams(self):
        #counts total number of ngrams
        a = 0
        for g in self.allgrams():
            a += 1
        return a
    
    def gotogram(self, gram, m=0,force = False):
        #returns a pointer to the tree for the desired n-gram
        current = self.freq
        for i in range(len(gram)):    
            #print (current[0])            
            c = gram[i]
            if c not in current[1].keys():
                if force:
                    current[1][c] = [0, {}]
                else:
                    raise ValueError("gram '" + gram + "' not found'" )

            current = current[1][c]
        if current[0] < m:
                
                raise ValueError("gram '" + gram + "' frequency too low" )
        return current

    
    def build_from_text(self, sentences):    
        #build tries from the given list of sentences.
        #if you have a lot of sentences, probably 
        #use an iterator to read through a file.
        
        for s in sentences:
            for k in range(self.numn+1):
                for i in range(len(s) - k+1):
                    gram = s[i: i +k]                    
                    self.gotogram(gram, force = True)[0] += 1
                    

                            
    def is_known(self, gram,m):
        #returns True if the given n-gram is in our list.
        try:
            self.gotogram(gram,m)
            return True
        except:
            return False

    def freqdict(self, gram,m=0):      
        #given an n-gram, returns a dictionary with the 
        #list of the characters that could possibly follow and their frequencies.

        ptr = self.gotogram(gram,m)                 
        return { c: ptr[1][c][0] for c in ptr[1].keys() if ptr[1][c][0] >=m}
        
 
    def allgrams(self, m=0, make = 'pointers',cleanup = False):
        #returns an iterator to look through all the known n-grams.
        # the string 'make' determines what the iterator should yield:
        # if 'none', don't yield anything.
        # if 'strings', yield the actual n-grams as strings.
        # if 'pointers', yield the sub-trees corresponding to the n-grams.        
        # if 'both', yield a pair (gram, ptr) where gram is the string and ptr is the tree.
    
        # if m is given, only yield the n-grams with frequency > m.
        # if cleanup == True, remove all subtrees corresponding to these rare n-grams.    
    
        
    
        
        self.indices = [-1]
        self.ptrs = []
        self.current = self.freq
        self.keylist = [ list(self.freq[1].keys())  ]
        gram = ''
        a = 0
        #lim = 1000
        
        while len(self.indices) > 0 :# and a < lim:
            a += 1
            #l = ptr[0]
            if self.indices[-1] == -1:
                #current node not yielded yet; yield 
                self.indices[-1] += 1
                #yield current
                #ptrs.append(current)
            
                #print (a, gram)
                #if self.current[0] >= m:
                if make == 'none':
                    pass
                elif make == 'strings':
                    yield gram
                elif make == 'pointers':
                    yield self.current
                elif make == 'both':
                    yield (gram, self.current)
                else:
                    raise ValueError("keyword " + make + "not found")
                
            elif self.indices[-1] < len(self.keylist[-1]):
                #indices[-1] += 1
                if self.current[1][self.keylist[-1][self.indices[-1]]][0] >= m:
                    self.ptrs.append(self.current)
                    self.current = self.current[1][self.keylist[-1][self.indices[-1]]]
                    gram += self.keylist[-1][self.indices[-1]]
                    self.indices[-1] += 1
                    self.indices.append(-1)
                    self.keylist.append(list(self.current[1].keys())    ) 
                else:
                    if cleanup:
                        self.current[1].pop( self.keylist[-1][self.indices[-1]] )
                    self.indices[-1] += 1
                
                
                #print (keylist)
                
                
                
                
            else:
                #done with branch. Pop up in tree
                if len(self.ptrs) > 0:
                    self.current = self.ptrs.pop()
                self.indices.pop()
                #print (keylist,'pop!')
                self.keylist.pop()
                gram = gram[:-1] 
 







def rand_unicode_char():
    #Generates a random unicode character from the basic multilingual plane.
    for i in range(10):
        X = 120737 * np.random.rand()
        if 0x0860 <= X <=  0x089F or 0x1C80 <= X <= 0x1CBF or 0x2FE0 <= X <= 0x2FEF:
            #these ranges are unallocated; ignore
            continue
        else:
            break
    #return unichr(int(X))
    try:
               
        return unichr(int(X))

    except:
        return rand_unicode_char()
    
def sample(A, problookup = None):
    #given a list A of w's, sample an entry w[0] with probability proportional to w[1].
    #or if dictionary, sample a key with probability proportional to A[key].
    if type(A) is dict:
        keys = list(A.keys())
        if problookup is None:
            b = 0
            problookup = []
            for key in keys:
                b += A[key]
                problookup.append(b)  
        X = np.random.rand() * problookup[-1]        
        i = min([i for i in range(len(problookup)) if problookup[i] > X    ])  
        #print(i)
        return keys[i]

    if problookup is None:
        b = 0
        problookup = []
        for i in A:
            b += A[-1][1]
            problookup.append(b)        
    X = np.random.rand() * problookup[-1]
    i = min([i for i in range(len(A)) if problookup[i] > X    ])    
    return A[i][0]


class languagemodel:
    #This is a basic language model based on n-gram frequency.

    def __init__(self, tries):
        self.p =  .75  #prob of ignoring a given recognized string in the current sentence of length k,
                      #and looking for a length k-1 string instead.
        self.pdumb = .001  #prob of completely ignoring information about the given sentence and generating
                           #a random unicode character instead.
        self.sentence = "" #current sentence consisting of observed characters thus far.
        self.numchars = 0  #the largest k to look for in k-grams.
        self.tries = tries #an 'ngramtries' class providing dictionary lookup for ngrams.
        
    def prob(self,sentence,c, minfreq = 0):
        #gives the probability of the next character being c given the history.
        p = self.p
        #self.numchars = 20
        pdumb = self.pdumb
        prob = 0
        m = -9
        

        if self.tries.is_known(c,minfreq):
            
            for k in range(min(self.numchars, len(sentence)+1), -1,-1):
                
                current = sentence[len(sentence) - k :]
                #print ('well...', k, current)
                #A = {key:self.ngrams[k][key] for key in self.ngrams[k].keys() if sentence[len(sentence)-k+1:] == key[:-1]}
                #if len(A) < 10:
                #    print(k,A)
                #if len(A) == 0:
                #print ( "trying...", current)
                #print("no ", k, "-match found; continuing.")
                #if current not in ngramsdict.keys():
                #    continue
                if len(sentence) < k or not self.tries.is_known(current,minfreq):
                    continue
                else:
                    if m == -9:
                        m = k
                #    m = k
                    #print (current, 'recognized!!')
                #if m == -9:
                #    m = k -1
                #if not ngrams.is_known(current) or c not in ngramsdict[current].keys():
                #if not self.tries.is_known(current + c):
                # print ("no match for ", current, "found; continuing.")
                #    continue
                #else:
                    #if m == -9:
                    #print (current, 'recognized!!',k)
                    #    m = k
                    #print (m,k)
                    #if m == -9:
                    #    m = k
                        
                if self.tries.is_known(current + c,minfreq):    
                    if k > 0 :
                        b = p
                    elif k == 0:
                        b = 1
                
                
                    #print (m,k, (1-p)**(m-k)*b ,current, self.tries.freqdict(current)[c], "/", self.tries.freqdict(current)) 
                    d = self.tries.freqdict(current,minfreq)
                    prob += ((1-p)**(m-k))*b *d[c]/sum(d.values())
               # print(current, k,prob)        
            #prob +=  (1-p)**(m-1)* float(self.ngrams[1][c]) / (sum([self.ngrams[1][key]  for key in self.ngrams[1].keys()]))
        return prob* (1 - pdumb) + pdumb/120737.0

    
    def testprob(self,sentence):
        #this tests that the probabilities sum to one.
        a = 0
        for c in self.tries.freqdict('').keys():
            pr = self.prob(sentence,c)
            
            a += pr
            print(c,pr, a)
            #if a > 1:
            #    print ("sum to more than one...")
                #break
        return a + pdumb/120737  * (120737 - len(self.tries.freqdict('').keys()))
        
    
    def logprob(self,sentence,c,m = 0):
        return np.log2(self.prob(sentence,c,m))

    def distribution(self):
        distrib = [(c, self.prob(c)) for c in self.ngrams[1].keys()]

        distrib.sort(key = lambda x: x[1])
        distrib.reverse()
        return distrib
        
    def generatechar(self):
        r = np.random.rand()
        if r < self.pdumb:
            c = rand_unicode_char()
            #print ('whoa')
            
            #, r
            #with probability pdumb, choose a uniformly random unicode character.
            #This ensures the probability of any string is positive.
        for k in range(min(self.numchars, len(self.sentence)+1), -1,-1):

            current = self.sentence[len(self.sentence)-k:]
            #A = {key:self.ngrams[k][key] for key in self.ngrams[k].keys() if self.sentence[len(self.sentence)-k+1:] == key[:-1]}
            #if current not in ngramsdict.keys():
            if not self.tries.is_known(current,m= 0):
                continue
            

            A = self.tries.freqdict(current)
            #print(k,current,ngramsdict[current])
            if len(A) == 0:
                #print("no match for", current, "found; continuing.")
                continue
                #if len(sentence) > len(startingword):
                #    print(sentence)
                #continue
            else:
                if np.random.rand() >= self.p and k > 0:
                    continue
                #print(A)
                c = sample(A)[-1]            
                break

        try:
            return c
        except:
            #print ("whaaaat")
            #print (c)
            return rand_unicode_char()
        
    def perplexity(self, sentence, m = 0):
        M = len(sentence)
        r = 0
        for i in range(M):
            #self.sentence = sentence[:i]
            
             
            q = self.logprob(sentence[:i],sentence[i],m)
            #print(sentence[:i], sentence[i], q  )
            r += q
        return 2**(-1.0/M * r)
    
    def generatestring(self, startingword, sentencelimit):       
        LM.sentence = startingword
        
        for i in range(sentencelimit) :
            c = LM.generatechar()
            LM.sentence = LM.sentence + c
            
            #if c in ['.', '!', '?']:
            #,    break   
            #sentence = sentence + c
            print(LM.sentence[-1])        
        print(LM.sentence)    
#
    
    
                #if c in ['.', '!', '?']:
                #    break 


#p = .75
#pdumb = .001
#
#sentencelimit = 5



mytries = np.load("finalngrams.npy").tolist()

#mytries = ngramtries(mytries)
#mytries.build_from_text(['y '])

LM = languagemodel(    mytries )
#print ('enter input')
LM.numchars = 4
LM.p = .6
LM.pdumb  = .01


#theinput = raw_input("Input string...").decode('utf-8')

theinput = ""

#print ('input lines!')  #remember to delete this line before submitting!!

while(True):
    try:
        #line = raw_input()

        line = raw_input().decode(sys.stdin.encoding or locale.getpreferredencoding(True))
        if line == "":
            break
        theinput += line + '\n'
    except:
        break
    



#
#i = 0
#sentence = ""
#
#totalp = 0

#LM.prob(' ')
#theinput = "o???o???o???q???goouonoiocooodoeoq.gx"

#file = open("test_input.txt", encoding="utf8")
#file = codecs.open( "foreign.txt", "r", "utf-8" )

#theinput = file.readline()
#file.close()
#theinput = "ohoeololooq.gx"


i = 0
LM.sentence = ""
while i < len(theinput):
    if theinput[i] == 'x':
        #print("quitting!")
        break
    elif theinput[i] == 'o':
        LM.sentence += theinput [i+1]
        print("character added!")
        if theinput[i+1] == '\u03':
            print("Stop character recognized. Clearing history.")
            LM.sentence = ""
        i += 2
    elif theinput[i] == 'g':        
        c = LM.generatechar()
        LM.sentence = LM.sentence + c
        print(c, "was generated!")
        i += 1
    elif theinput[i] == 'q':
        c = theinput[i+1]
        #print("log probability of character", c,"is", LM.logprob(c))
        print(LM.logprob(LM.sentence, c))
        i += 2
        
    else:
        print("unexpected input! Do not recognize", theinput[i],"of", theinput, ". Quitting...")
        break
