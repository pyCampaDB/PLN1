from nltk import download as nltkDownload

nltkDownload('punkt')
nltkDownload('stopwords')
from requests import get as getReq
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords #empty words
from nltk.stem import SnowballStemmer
from collections import Counter
from wordcloud import WordCloud
from matplotlib.pyplot import (figure, imshow as pltImshow, axis, show as pltShow,
                               bar, xlabel, ylabel, xticks, tight_layout, title as plttitle)



######################################################### METHODS ################################################################################################
def read_and_processing(url):
    #Download the URL's content
    response = getReq(url)
    content_response = response.content

    #Extract the text into the content
    content_format = BeautifulSoup(content_response, 'html.parser')
    text = content_format.get_text()

    #Process the text
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalpha()] #We only want to extract the words
    empty_words = set(stopwords.words('spanish')) #Load the Spanish's empty words
    words = [word for word in words if word not in empty_words]
    bag_of_words = Counter(words) #Build the frequency dictionary
    print(bag_of_words.most_common(10)) #Print the 10 most frequenly words

    return bag_of_words, words

def run():
    #Check us the function execute successfully with the Quijote's URL
    bag_words_quijote, words_quijote = read_and_processing('https://www.gutenberg.org/cache/epub/2000/pg2000')
    print()
    text_quijote = ' '.join(words_quijote)
    #The number of times the word 'rocinante' appears
    print(bag_words_quijote['rocinante'])
    print()
    #Create the WordCloud Object
    wordcloud= WordCloud(width=800, height=400,
                         background_color='white'
                         ).generate(text_quijote)
    
    #Print the cloud of words
    figure(figsize=(10,5))
    pltImshow(wordcloud, interpolation='bilinear')
    axis('off')
    pltShow()

    #Other kind of display
    bag_words_sorted = Counter(dict(sorted(bag_words_quijote.items(),
                                           key= lambda item: item[1], reverse=True)))
    words = list(bag_words_sorted.keys())[0:15]
    frequencies = list(bag_words_sorted.values())[0:15]

    figure(figsize=(10,6))
    bar(words, frequencies, color='skyblue')
    xlabel('Word')
    ylabel('Frequency')
    plttitle('Bag of Words Quijote')
    xticks(rotation=45)
    tight_layout()
    pltShow()


############################################################# MAIN #######################################################################################
if __name__ == '__main__':
    run()