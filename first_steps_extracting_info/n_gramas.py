#n-grama 
from nltk import download as nltkDownload
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from requests import get as getReq
from bs4 import BeautifulSoup

######################################################## METHODS ##########################################################
def calculate_n_grams(text, n):
    words = word_tokenize(text)
    n_grammes = list(ngrams(words, n))
    return n_grammes



def read_and_preprocess(url):   
    response = getReq(url)
    content_response = response.content

    content_format = BeautifulSoup(content_response, 'html.parser')
    text = content_format.get_text()
    return text

def run():
    text = "Este es un ejemplo de texto para calcular n-gramas en Python basándonos en NLTK"

    n = int(input('Enter the size of n-gramme: '))
    result = calculate_n_grams(text, n)
    print(f'{n}-grammes of text: ')
    for gram in result:
        print(gram)

    words_quijote = read_and_preprocess(
        'https://www.gutenberg.org/cache/epub/2000/pg2000.txt'
        )
    result_quijote = calculate_n_grams(words_quijote, n)
    print()
    print()
    print(f'{n}-grammes of text:\n')
    cont = 0
    for gram in result_quijote:
        print(gram)
        cont += 1
        if cont > 20:
            break
    
###################################################3 MAIN ##################################################################3333
if __name__ == '__main__':
    run()