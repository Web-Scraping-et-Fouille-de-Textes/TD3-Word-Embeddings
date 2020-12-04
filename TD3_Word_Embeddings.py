#!/usr/bin/env python
# coding: utf-8

# # TD3 Word Embeddings
# ## Exercice n°1 : Constitution des sous-corpus

# In[2]:


import io

"""
# Exemple ouverture d'un fichier pour lire
inFilePath = 'file.txt'
inFile = io.open(inFilePath, mode='r', encoding='utf-8')
line = inFile.readlines()
inFile.close()

# Exemple ouverture d'un fichier pour ecrire
outFilePath = 'file.txt'
s = "Ceci est un enonce."
outFile = io.open(outFilePath, mode='r', encoding='utf-8')
outFile.write(s)
outFile.close
"""

gauche, droite, parti = '', '', ''
corpusPath = './content/HYPERBASE_Droite_VS_Gauche.txt'

inFile = io.open(corpusPath, mode='r', encoding='utf-8')
line = inFile.readline()
while line != '' :
  if '****' in line : # ligne de métadonnées
    if '*parti_gauche' in line :
      parti = 'gauche'
    if '*parti_droite' in line :
      parti = 'droite'
  else :
    if line != '\n' and parti != '' :
      if parti == 'droite' :
        droite += line
      else :
        gauche += line

  line = inFile.readline()

inFile.close()

print(" ===== Partie droite ===== ")
print(droite[:100])

print("\n ===== Partie gauche =====")
print(gauche[:100])


# ## Exercice n°2 : Les embeddings
# 
# D'après l'article d'OpenClassRoom (https://tinyurl.com/yxaghuvg) expliquer les termes suivant :
# - distributional hypothesis : Correspond au faites de prendre le "contexte" autour du mot sélectionné afin d'avoir un ensemble de valeurs (mots) souvent utilisé avec le terme choisi.
# - représentation vectorielle : représentation (possiblement graphique) qui permet de trouver la distance entre un terme et un autre appelé vecteur. Cela permet de faire des représentations male-femelle par exemple.
# - translation linéaire : Action qui permet de passer d'un terme à un autre en suivant une ligne directrice.
# - vecteur dense
# - Continuous Bag of Words(CBOW) : Méthode d'entrainement d'un réseau de neurones afin de prédire un mot avec un context défini
# - skip-gram : Par oppostion au CBOW, c'est une méthode d'entrainement d'un réseau de neurones qui prédit le context d'un mot
# 
# ## Exercice n°3 : D´efinition d’une classe Embeddings
# 
# Pour le fichier ***./content/embeddings.py***, sur la class *Embeddings* :
# - Quelle librairie est utilisée pour procéder à l'apprentissage des embeddings ? gensim.models
# - Quel modèle en particulier ? Word2Vec
# - Décriver le processus d'acquisition des données textuelles. En particulier, quels biais voyez-vous ? Proposez, sans les réaliser, deux améliorations possibles.
# 
# ## Exercice n°4 : Apprentissage des embeddings
# 

# In[3]:


import sys  
sys.path.insert(0, './content')
import embeddings as emb

def writeDocument(path, content) :
    outFile = io.open(path, mode='w+', encoding='utf-8')
    outFile.write(content)
    outFile.close

# On définie les path des futur fichiers    
outFileDroite = "./content/outFileDroite.txt"
outFileGauche = "./content/outFileGauche.txt"

# On écrit les informations
writeDocument(outFileDroite, droite)
writeDocument(outFileGauche, gauche)

# Path de save des modeles
outFileModelDroite = "./content/modelDroite.W2Vmodel"
outFileModelGauche = "./content/modelGauche.W2Vmodel"

# On init
embeddingsDroite = emb.Embeddings(outFileDroite, outFileModelDroite)
embeddingsGauche = emb.Embeddings(outFileGauche, outFileModelGauche)

# Learn
embeddingsDroite.learn()
embeddingsGauche.learn()

word = "présidente"
print(" ===== Droite =====")
print(embeddingsDroite.get_vector(word))
print(embeddingsDroite.get_vocab())

print("\n\n ===== Gauche =====")
print(embeddingsGauche.get_vector(word))
print(embeddingsGauche.get_vocab())


# ## Exercice n°5 : Utilisation des embeddings
# 
# ### 1. Etude des vecteurs les plus proches et éloignés

# In[4]:


import numpy as np

def cosine_similarity (vec1 , vec2 ):
    dot = np.dot(vec1 , vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cos = dot / (norm1 * norm2)
    return cos 

vocabDroite, vocabGauche = embeddingsDroite.get_vocab(), embeddingsGauche.get_vocab()

sims = {}
for wordItem in vocabDroite :
    if wordItem in vocabGauche :
        vec1 = embeddingsDroite.get_vector(wordItem)
        vec2 = embeddingsGauche.get_vector(wordItem)
        sims[wordItem] = cosine_similarity(vec1, vec2)
        
sims_up = {k:v for k,v in sorted(sims.items (), key=lambda i:i[1], reverse=True )}
sims_down = {k:v for k,v in sorted(sims.items (), key=lambda i:i[1])}

print('∗∗∗ Vecteurs proches : ')
i = 0

while i < 30 :
    w = list( sims_up .keys ())[i]
    print(w, end=', ')
    i += 1

print('\n\n∗∗∗ Vecteurs éloignés : ')
i = 0
while i < 30 :
    w = list( sims_down .keys ())[i]
    print(w, end=', ')
    i += 1     


# On peut constater que ce sont pour la plupart des verbes avec peut de stop-word ou d'erreur (comme le ma ou -).
# 
# ### 2. Observation de vecteurs similaires

# In[5]:


words = ['présidente', 'patrie', 'amour']
topN = 10
corpus = 'Droite'
for word in words :
    sims = ' '.join ([w for w,s in embeddingsDroite.most_similar(word, topN )])
    print('\n{}\t{}\t{}'.format(word, corpus, sims ))

print("\n\n ===== ----- ===== \n\n")
    
corpus = 'Gauche'
for word in words :
    sims = ' '.join ([w for w,s in embeddingsGauche.most_similar(word, topN )])
    print('\n{}\t{}\t{}'.format(word, corpus, sims ))


# On peut constater que les réponse sont plutôt cohérente avec les mots choisis. 
# Mais le sous-corpus Gauche est beaucoup plus large sur les mots choisis.
# 
# ### 3. Analogies
# 
# Testez quelques analogies sur les deux sous-corpus. Pour commencez :
# - président - homme + femme
# - candidat - homme + femme
# - député - candidat + candidats

# In[6]:


def most_sim(pos, neg, embeddings):
    most_sim = embeddings.most_similar_analogy (pos, neg, topN = 5)
    for w_s in most_sim :
        word, sim = w_s
        print('\n{}\t{}'.format(word , sim ))
        
pos, neg = ['roi', 'femme'], ['homme']
print("===== Droite =====")
most_sim(pos, neg, embeddingsDroite)
print("\n\n===== Gauche =====")
most_sim(pos, neg, embeddingsGauche)

pos, neg = ['présidente', 'femme'], ['homme']
print("\n\n===== Droite =====")
most_sim(pos, neg, embeddingsDroite)
print("\n\n===== Gauche =====")
most_sim(pos, neg, embeddingsGauche)

pos, neg = ['candidat', 'femme'], ['homme']
print("\n\n===== Droite =====")
most_sim(pos, neg, embeddingsDroite)
print("\n\n===== Gauche =====")
most_sim(pos, neg, embeddingsGauche)

pos, neg = ['député', 'candidats'], ['candidat']
print("\n\n===== Droite =====")
most_sim(pos, neg, embeddingsDroite)
print("\n\n===== Gauche =====")
most_sim(pos, neg, embeddingsGauche)


# En fonction du sous-corpus le résultat difère même si certains reste relativement proche.
# 
# ## Exercice n°6 : DEVOIR 
# 

# In[13]:


from sklearn.manifold import TSNE
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plot(plot_title, model) :
    labels, tokens = [], []
    print("----- step 1 -----")
    for word in model.wv.vocab :
        tokens.append(model[word])
        labels.append(word)
        #print(word)

    print("----- step 2 -----")
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    print("----- step 3 -----")
    new_values = tsne_model.fit_transform(tokens)
    
    print("----- step 4 -----")
    x, y= [], []
    for value in new_values :
        x.append(value [0])
        y.append(value [1])
    
    plt.figure(figsize =(16 , 16))
    print("----- step 5 -----")
    print(" > "+str(len(x))+' éléments')
    for i in range(len(x)) :
        #print('label : '+labels[i])
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
        xy=(x[i], y[i]),
        xytext =(5, 2),
        textcoords ='offset points', ha='right', va='bottom')
    plt.title(plot_title)
    plt.show()
    
embeddingsDroite.learn_restrictive(1500)
embeddingsGauche.learn_restrictive(1500)

tsne_plot('test', embeddingsDroite.get_model())

