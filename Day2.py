
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize,PunktSentenceTokenizer
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk import pos_tag


txt = "Mary jumps in a field and following her Sam also jumped. That our lives would be changed forever. The world was loud with carnage and sirens and then quiet with missing voices that would never be heard again. These lives remain precious to our country and infinitely precious to many of you. Today, we remember your loss, we share your sorrow, and we honor the men and women you have loved so long and so well. For those too young to recall that clear September day, it is hard to describe the mix of feelings we experienced."
#txt = "Mary jumps in a field and following her Sam also jumped. That our lives would be changed forever."

sentence=sent_tokenize(txt)
print(sentence)

words=word_tokenize(txt)
print(words)

pos=pos_tag(words)
print(pos)

#for sent in sentence:
#    print(pos_tag(word_tokenize(sent)))

res=[]
ps=PorterStemmer()
list1 = ["jumps","jumped"]
for w in list1:
   #print(ps.stem(w))
   res.append(ps.stem(w))
print(res)

lem=WordNetLemmatizer()
list2 = ['voices', 'sirens']
lm = [lem.lemmatize(word) for word in list2]
print(lm)


