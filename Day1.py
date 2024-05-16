
# With respect to the sample texts provided below, practice word and sentence tokenization.
# Also create a list of filtered words using stop word library in Nltk. You can also create a list of your own stop words to get more useful word

import nltk
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

s1="20 years ago, we all found in different ways in different places but all at the same moment."
s2="That our lives would be changed forever. The world was loud with carnage and sirens and then quiet with missing voices that would never be heard again."
s3="These lives remain precious to our country and infinitely precious to many of you. Today, we remember your loss, we share your sorrow, and we honor the men and women you have loved so long and so well. For those too young to recall that clear September day, it is hard to describe the mix of feelings we experienced. "
s4="There was horror at the scale of destruction. and awe at the bravery and kindness that rose to meet it. There was, shock! at the audacity of evil and gratitude, for the heroism and decency that opposed it? In the sacrifice!"

#txt=s1+s2+s3+s4

txt=s1+s2

sep_words= word_tokenize(txt)
sep_sent=sent_tokenize(txt)
# type(sep_words)
print(sep_words)
#print(sep_sent)

#stop_words=set(stopwords.words('english'))
stop_words=["in",".",";","?",","]
filter_sent=[w for w in sep_words if not w in stop_words]
print(filter_sent)
print(len(sep_words))
print(len(filter_sent))


