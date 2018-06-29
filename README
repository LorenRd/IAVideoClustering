
RESUMEN DE VIDEO MEDIANTE CLUSTERING

*Estructura del codigo fuente
	+Funciones
		-Principal()
			Parametros K,T y H
			Crea directorios, extrae frames del video, llama a CalcularFotogramasClave(directorio,K,T,H)
			para obtener ListaFramesConClasificacion y ListaKeyFrames
			Una vez aplicado el algoritmo imprime los KeyFrames y genera el video resumen.
		-CalcularFotogramasClave(directorio,K,T,H)
			Genera ListaFrames con el valor medio RGB y los nombres de las capturas, luego llama a AplicaKmedias y CalculaCentroidesClaves
		-AplicaKmedias(ListaFrames,K)
			Aplica KMedias a las capturas y devuelve ListaFramesConClasificacion
		-CalculaCentroidesClaves(ListaFramesConClasificacion, K)
			Devuelve los KeyFrames que son las capturas mas proximas a los centroides
		-dist(a, b, ax=1)
			Devuelve la distancia Manhattan para los centroides
		-ExtraeFrames(T,directorio)
			Extrae los frames de un video
		-CreaVideo(directorio)
			Genera un video a partir de los frames

*Uso
	Fija los valores K,T y H ademas de los directorios de entrada y salida:
		K = 6
		T = 1
		H = 256
		directorio="INPUTPATH"
		destino="OUTPUTPATH"

	Para ejecutar introducir por consola desde el terminal estando en el directorio de proyecto.py:
		python proyecto.py

	Introduce el nombre del video a resumir y empezara a funcionar el script.
