from warnings import filterwarnings
filterwarnings('ignore')

from pandas import read_csv


def jump():
    print()
    print()


info_hotels = read_csv('real_practices/hoteles.csv')
print(info_hotels.head())
jump()
print(info_hotels.shape)
jump()
info_hotels_without_neutrals = info_hotels[info_hotels.label!=1]
from matplotlib.pyplot import  show, xlabel, ylabel

info_hotels_without_neutrals['label'].plot(
    kind='hist',
    bins=20,
    title='label'
)
xlabel('Empty...')
ylabel('Frequency')
show()

title_feel = info_hotels_without_neutrals[['title', 'label']]

from nltk import download as nltkdownload
nltkdownload('punkt')
nltkdownload('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def preprocess(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalpha()]
    empty_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in empty_words]
    #End load data and preprocess
    return words
title_feel['titulo_preprocesado'] = title_feel['title'].apply(lambda text: preprocess(text))
jump()
print(title_feel.head())


def evaluate_title(title_preprocess, list_positive_words, list_negative_words):
    result = 0
    for word in title_preprocess:
        if word in list_negative_words:
            result -= 1
        if word in list_positive_words:
            result += 1
    return result

positive_words = ['bueno', 'genial', 'estupendo', 'buen',
                  'bien', 'fantástico', 'lujo', 'agradable', 
                  'maravilloso', 'espectacular', 'increible',
                  'aceptable', 'contento', 'espectacular']
negative_words = ['malo', 'mal', 'flojo', 'fatal', 'terrible',
                  'feo', 'lejos', 'ruido', 'infame']

title_feel['prediccion_basica'] = title_feel['titulo_preprocesado'].apply(lambda text: evaluate_title(text, positive_words, negative_words))
jump()
print(title_feel.head())
jump()

print(title_feel[title_feel.prediccion_basica!=0].shape[0])
jump()
print(title_feel[title_feel.prediccion_basica!=0].shape[0]/title_feel.shape[0])
jump()

print(title_feel[title_feel.prediccion_basica!=0])

def tunning_predict(prediction):
    if prediction < 0:
        return 0
    elif prediction > 0:
        return 3
    else:
        return 2
    
title_feel['prediccion_ajustada'] = title_feel['prediccion_basica'].apply(lambda prediction: tunning_predict(prediction))
print(title_feel[title_feel.prediccion_basica!=0].head())
jump()
predict_emited = title_feel[title_feel.prediccion_basica!=0]
print(predict_emited.shape[0])
jump()
predict_true = predict_emited[predict_emited["label"]==predict_emited["prediccion_ajustada"]]
print(predict_true.shape[0])
jump()
predict_failed = predict_emited[predict_emited['label']!=predict_emited['prediccion_ajustada']]
jump()
predict_optimistas = predict_failed[predict_failed["prediccion_ajustada"]==3]

print(predict_optimistas.head())
jump()
print(predict_optimistas.shape[0])
jump()
predict_failed = predict_failed[predict_failed["prediccion_ajustada"]==0]
print(predict_failed.head())
jump()
print(predict_failed.shape[0])

from sentiment_analysis_spanish import sentiment_analysis
model_sentiment = sentiment_analysis.SentimentAnalysisSpanish()

def feel(text, sentiment=model_sentiment):
    return sentiment.sentiment(text)
jump()
print(feel('esto es genial'))
jump()
print(feel('esto es horrible'))
jump()
title_feel['prediccion_ia'] = title_feel['title'].apply(lambda text: feel(text))
print(title_feel['prediccion_ia'].head())
def tunning_predict_ia(predict):
    if predict < 0.5:
        return 0
    else:
        return 3

title_feel['prediccion_ia_ajustada'] = title_feel['prediccion_ia'].apply(lambda predict: tunning_predict_ia(predict))

predict_true_ia = title_feel[title_feel['label']==title_feel['prediccion_ia_ajustada']]
jump()
print('predict_true_ai.shape[0]')
print(predict_true_ia.shape[0])
jump()
print('predict_true_ai.shape[0]/title_fell.shape[0]')
print(predict_true_ia.shape[0]/title_feel.shape[0])
