from nltk import download as nltkdownload
nltkdownload()

from nltk import corpus

nltkdownload('gutenberg')

corpus.gutenberg.fileids()

don_quijote="""
En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas, con sus pantuflos de lo mesmo, y los días de entresemana se honraba con su vellorí de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años. Era de complexión recia, seco de carnes, enjuto de rostro, gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de «Quijada», o «Quesada», que en esto hay alguna diferencia en los autores que deste caso escriben, aunque por conjeturas verisímiles se deja entender que se llamaba «Quijana»,. Pero esto importa poco a nuestro cuento: basta que en la narración dél no se salga un punto de la verdad.

Es, pues, de saber que este sobredicho hidalgo, los ratos que estaba ocioso —que eran los más del año—, se daba a leer libros de caballerías, con tanta afición y gusto, que olvidó casi de todo punto el ejercicio de la caza y aun la administración de su hacienda; y llegó a tanto su curiosidad y desatino en esto, que vendió muchas hanegas de tierra de sembradura para comprar libros de caballerías en que leer, y, así, llevó a su casa todos cuantos pudo haber dellos; y, de todos, ningunos le parecían tan bien como los que compuso el famoso Feliciano de Silva, porque la claridad de su prosa y aquellas entricadas razones suyas le parecían de perlas, y más cuando llegaba a leer aquellos requiebros y cartas de desafíos, donde en muchas partes hallaba escrito: «La razón de la sinrazón que a mi razón se hace, de tal manera mi razón enflaquece, que con razón me quejo de la vuestra fermosura». Y también cuando leía: «Los altos cielos que de vuestra divinidad divinamente con las estrellas os fortifican y os hacen merecedora del merecimiento que merece la vuestra grandeza...»

Con estas razones perdía el pobre caballero el juicio, y desvelábase por entenderlas y desentrañarles el sentido, que no se lo sacara ni las entendiera el mesmo Aristóteles, si resucitara para solo ello. No estaba muy bien con las heridas que don Belianís daba y recebía, porque se imaginaba que, por grandes maestros que le hubiesen curado, no dejaría de tener el rostro y todo el cuerpo lleno de cicatrices y señales. Pero, con todo, alababa en su autor aquel acabar su libro con la promesa de aquella inacabable aventura, y muchas veces le vino deseo de tomar la pluma y dalle fin al pie de la letra como allí se promete; y sin duda alguna lo hiciera, y aun saliera con ello, si otros mayores y continuos pensamientos no se lo estorbaran. Tuvo muchas veces competencia con el cura de su lugar —que era hombre docto, graduado en Cigüenza— sobre cuál había sido mejor caballero: Palmerín de Ingalaterra o Amadís de Gaula; mas maese Nicolás, barbero del mesmo pueblo, decía que ninguno llegaba al Caballero del Febo, y que si alguno se le podía comparar era don Galaor, hermano de Amadís de Gaula, porque tenía muy acomodada condición para todo, que no era caballero melindroso, ni tan llorón como su hermano, y que en lo de la valentía no le iba en zaga.
"""

nltkdownload('punkt')
from nltk.tokenize import word_tokenize
words = word_tokenize(don_quijote)
print("\n"+str(type(words))+"\n")
print("\n"+str(words[0:20])+"\n")

words_lowercase =[word.lower() for word in words]
print("\n"+str(words_lowercase[0:20])+"\n")

real_words = [word for word in words if word.isalpha()]
print("\n"+str(real_words[0:20])+"\n")


nltkdownload('stopwords')
from nltk.corpus import stopwords
empty_words = set(stopwords.words('spanish'))
print("\n"+str(list(empty_words)[0:20])+"\n")

print(f"\nCantidad de palabras vacías: {len(empty_words)}\n")

highligh_words = [word for word in real_words if word not in empty_words]
print(f"\nPalabras más destacadas: {str(highligh_words)}\n")

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')
root_words= [stemmer.stem(word) for word in highligh_words]
print(f"\nPalabras raíces: {str(root_words[0:20])}\n")

from spacy import load as spacyLoad

model_sp = spacyLoad("es_core_news_sm")
don_quijote_sp=model_sp(don_quijote)

#Análisis morfológico del texto
contador = 0
for token in don_quijote_sp:
  print(token.text, token.pos_)
  contador= contador + 1
  if contador > 30:
    break
  
#Extracción de entidades
for ent in don_quijote_sp.ents:
  print(ent.text, ent.label_)

"""
Observamos que el modelo acierta en muchas ocasiones, pero en otras muchas 
no acaba de afinar. Es importante entender también la dificultad del texto 
por su antigüedad. Los modelos tienden a funcionar mejor cuando el lenguaje
 es más popular:
"""
example_sentence = model_sp("Donald Trump y Joe Biden se batirán en los próximos comicios de la ciudad de Nueva York por el gobierno de Estados Unidos.")
for ent in example_sentence.ents:
  print(ent.text, ent.label_)