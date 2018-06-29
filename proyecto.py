#!/usr/bin/python
#Imports
import os, sys
import cv2
import pandas as pd
import numpy as np
import glob #Ficheros
import shutil #Directorios
from copy import deepcopy
from matplotlib import pyplot as plt

def Principal():
	#Variable K,T,H

	#K -> En K medias el numero de centros, numero de KeyFrames a extraer
	K = 3
	#T -> Numero de fotogramas a saltar, en funcion de la longitud del video y del numero de fotogramas sera mayor o menos
	T = 1 #Vamos a extraer 1 frame por segundo de video
	#H -> Tamano del histograma, 256 porque el histograma rgb va de 0 a 255 (0 mas oscuro y 255 el mas intenso)
	H = 256

	directorio="INPUTPATH"
	destino="OUTPUTPATH"

	#Borramos y creamos si no existia el INPUTPATH
	if os.path.exists(directorio):
	    shutil.rmtree(directorio)
	os.makedirs(directorio)

	#Extraemos frames del video
	ExtraeFrames(T,directorio)

	#Calculamos los fotogramas clave
	ListaFramesConClasificacion, ListaKeyFrames=CalcularFotogramasClave(directorio,K,T,H)

	#Copiamos los KeyFrames en el OUTPUTPATH
	imagenes = list(sorted(glob.glob(os.path.join(directorio,'*.*'))))
	if os.path.exists(destino):
	    shutil.rmtree(destino)
	os.makedirs(destino)
	for i in range (len(ListaKeyFrames)):
		print(ListaKeyFrames[i])
		print(imagenes[ListaKeyFrames[i]+1])
		shutil.copy2(imagenes[ListaKeyFrames[i]+1], destino)

	#Crear video
	CreaVideo(destino)

def CalcularFotogramasClave(directorio,K,T,H):
	#Nombre de las ficheros capturas
	Frames=[]
	#Media entre los 3 canales (rgb) para cada valor de la intensidad (0-255)
	Medias=[]

	#Recorremos todas los frames extraidos del video
	for image in os.listdir(directorio):
		input_path = os.path.join(directorio,image)
		#Anadimos los nombres de las capturas a la lista
		Frames.append(input_path)
		img = cv2.imread(input_path)
		#Si queremos mostrar la imagen en cuestion
		#cv2.imshow('image',img)

		#[0],[1],[2] Canales azul,verde o rojo respectivamente
		histrB = cv2.calcHist([img],[0],None,[H],[0,H])
		histrG = cv2.calcHist([img],[1],None,[H],[0,H])
		histrR = cv2.calcHist([img],[2],None,[H],[0,H])
		#calcHist devuelve un array de 256 valores
		#Cada valor corresponde al numero de pixeles con esa intensidad en cuestion en un determinado canal

		#print(histrB[0][0]) #Numero de pixeles azules con intensidad 0
		MediaFotograma=[]
		for i in range(0,H):
			MediaFotograma.append((histrB[i][0]+histrG[i][0]+histrR[i][0])/3)
		Medias.append(MediaFotograma)
		#Media es nuestro vector de H valores de cada fotograma
		#print(Media[0]) Media de pixeles entre los 3 canales con intensidad 0

	#Array bidimensional con nombres y valores medios de cada captura
	ListaFrames=[[],[]]
	for i in range (0,len(Frames)):
		ListaFrames[0].append(Frames[i])
		ListaFrames[1].append(Medias[i])

	#Aplicamos KMedias y obtenemos una lista de puntos clasificados segun su centroide
	ListaFramesConClasificacion=AplicaKmedias(ListaFrames,K)
	#Los KeyFrames los obtenemos con la captura mas proxima a un centroide
	ListaKeyFrames=CalculaCentroidesClaves(ListaFramesConClasificacion,K)

	return ListaFramesConClasificacion, ListaKeyFrames

def AplicaKmedias(ListaFrames,K):
	#Para cada captura vamos a extraer un unico valor medio RGB, sera el mas repetido en todos sus pixeles
	ValorMedioRGBFrames=[]
	for i in range(0,len(ListaFrames[0])):
		ValoresRGB = []
		ValoresRGB=ListaFrames[1][i]
		ValorMedioRGBFrames.append(ValoresRGB.index(max(ValoresRGB)))
	#print(ValorMedioRGBFrames)

	#Aplicamos K Medias
	plt.rcParams['figure.figsize'] = (16, 9)
	plt.style.use('ggplot')
	#En el eje x: 0 - N de capturas 
	#En el eje y: 0 - 255
	data = pd.DataFrame({
		'x':[i for i in range(0,len(ListaFrames[0]))],
		'y':[ValorMedioRGBFrames[j] for j in range(0,len(ValorMedioRGBFrames))]	
		})
	f1 = data['x'].values
	f2 = data['y'].values
	X = np.array(list(zip(f1, f2)))
	plt.scatter(f1, f2, c='black', s=7)
	#Pintamos los puntos de cada captura en el plano
	#plt.show()
	#Centroides
	#Cordenada X aleatoria entre 0 y numero de frames
	C_x = np.random.randint(0, len(ValorMedioRGBFrames), size=K)
	#Cordenada Y aleatoria entre 0 y 255
	C_y = np.random.randint(0, np.max(X), size=K)
	C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
	plt.scatter(f1, f2, c='#050505', s=7)
	plt.scatter(C_x, C_y, marker='*', s=200, c='g')
	#Pintamos los centroides
	#plt.show()
	#Actualizamos los valores de los centroides por cada iteracion
	C_old = np.zeros(C.shape)
	clusters = np.zeros(len(X))
	#Error de los centroides calculado por la distancia Manhattan Norma(a-b)
	error = dist(C, C_old, None)
	#Suma de los errores para calcular K optimo (Elbow Method)
	SumErrores=0 
	#Itera hasta que el error sea 0
	while error != 0:
	    #Asigna cada valor a su cluster mas cercano
	    for i in range(len(X)):
	        distances = dist(X[i], C)
	        cluster = np.argmin(distances)
	        clusters[i] = cluster
	    #Guarda los valores antiguos de los centroides
	    C_old = deepcopy(C)
	    #Encuentra los nuevos centroides calculando el valor medio
	    for i in range(K):
	        points = [X[j] for j in range(len(X)) if clusters[j] == i]
	        C[i] = np.mean(points, axis=0)
	    error = dist(C, C_old, None)
	    SumErrores += (error*error)
	#print(SumErrores)
	#Lista de colores para los puntos de cada cluster
	colors = ['#030303','#DAA520','#228B22','#FF3030','#1E90FF','#FF1493','#C1FFC1','#BF3EFF','#CAFF70','#3D59AB','#7FFF00','#E3CF57','#00FFFF']
	fig, ax = plt.subplots()
	#Array bidimensional con el cluster y los puntos pertenecientes a cada uno
	ListaFramesConClasificacion=[[],[]]
	for i in range(K):
		#print ("Cluster: "+str(i+1))
		points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
		ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
		#print(points)
		#Anadimos los puntos agrupados por clusters
		ListaFramesConClasificacion[1].append(list(points[i][0] for i in range(len(points))))
	#Anadimos centroides
	ListaFramesConClasificacion[0].append(C)
	ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
	#Grafico con KMedias aplicado	
	#plt.show()
	return ListaFramesConClasificacion

#Un KeyFrame por cada centroide (La captura mas proxima al centroide)
def CalculaCentroidesClaves(ListaFramesConClasificacion, K):
	ListaKeyFrames=[]
	for i in range(0,K):
		#Redondeamos la cordenada X del centroide para sacar la captura mas proxima a el
		ListaKeyFrames.append(int(round(ListaFramesConClasificacion[0][0][i][0])))

	return ListaKeyFrames


# Calcula distancia Manhattan
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

#Extrae frames
def ExtraeFrames(T,directorio):
	#Pedir nombre del video
	video = raw_input("Nombre del video: ")
	if T==1:
		os.system("ffmpeg -i "+str(video)+" -vf fps=1 "+directorio+"/captura%04d.jpg")
	else:
		os.system("ffmpeg -i "+str(video)+" -vf fps=1/"+str(T)+" "+directorio+"/captura%04d.jpg")

#Crea video resumen a partir de los KeyFrames
def CreaVideo(directorio):
	os.system("ffmpeg -framerate 1 -pattern_type glob -i '"+directorio+"/*.jpg' -vcodec libx264 "+directorio+"/resumen.mp4")

if __name__ == "__main__":
	Principal()