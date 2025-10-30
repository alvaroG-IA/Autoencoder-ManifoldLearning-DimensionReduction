# Autoencoder-ManifoldLearning-DimensionReduction

Repositorio de proyecto para explorar técnicas de **reducción de dimensión** mediante autoencoders y aprendizaje de variedades (manifold learning).

## 📚 Contenido
Este proyecto incluye:

- Código en Python para entrenar un **autoencoder** que reduce la dimensión de datos de alta dimensión.  
- Implementaciones de técnicas de **manifold learning** (por ejemplo, t-SNE, UMAP, Isomap) para comparar con la representación latente aprendida.  
- Visualizaciones de la dimensión reducida y análisis comparativo de métodos.  
- Archivo `requirements.txt` con dependencias del proyecto.

## 🎯 Objetivos del proyecto
- Construir y entrenar un autoencoder para aprendizaje de representaciones latentes.  
- Aplicar y comparar métodos de manifold learning para análisis de datos de alta dimensión.  
- Evaluar cómo la estructura latente capturada por el autoencoder se compara con la estructura preservada por técnicas clásicas de reducción de dimensión.  
- Proveer un recurso reutilizable para trabajos de IA, visión por computadora, procesamiento de señales, etc.

## ⚙️ Instalación
1. Clona el repositorio:  
   ```bash
   git clone https://github.com/alvaroG-IA/Autoencoder-ManifoldLearning-DimensionReduction.git
   cd Autoencoder-ManifoldLearning-DimensionReduction
   ```
2. Crea y activa el entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Uso rápido

   ```bash
   python mixed_manifol_detector.py conjunto_train.csv conjunto_test.csv
   ```

Remplaza `conjunto_train.csv` por la ruta del documento `.csv` que desas utilizar para entrenar el conjunto y `conjunto_test.csv`por la ruta del conjunto de test.

## 📁 Estructura del proyecto

   ```
   Autoencoder-ManifoldLearning-DimensionReduction/
   │
   ├── README.md
   ├── requirements.txt
   ├── mixed_manifold_detector.py
   │
   └── packages/
       ├── Utils.py
       ├── MixedManifoldDetector.py
       └── autoencoders/
           ├── Autoencoder.py
           ├── LinearAutoencoder.py
           ├── LinearSparseAutoencoder.py
           └── DenoisingSparseAutoencoder.py
   ```

- En el script `mixed_manifold_detector.py` encontrarás la logica seguida para ejecutar el sistema.
- Dentro del directorio `packges/` encontraremos:
   - `Utils.py`: módulo que recopila diferentes funciones usadas durante la ejecución.
   - `MixedManifoldDetector.py`: clase principal en la que se desarrolla el sistema.
   - `autoencoders/`: directorio que recopila todas las implementaciones de los diferentes autoencoders utilizados.
      - `Autoencoder.py`: interfaz del resto de autoencoders.
      - `LinearAutoencoder.py`: implementación de la arquitectura de un autoencoder lineal.
      - `LineasSparseAutoencoder`: implementación de autoencoder lineal con regularización sparse (L1)
      - `DenoisingSparseAutoencoder`: implementación de autoencoder lineal con tanto regularización sparse como con regularización denoising.

## 📄 Licencia
Este proyecto está licenciado bajo la Licencia MIT. Revisar el archivo LICENSE para más detalles (si lo incluyes).

## 📨 Contacto
Álvaro García Velasco
- e-mail: alvarogarciavelasco1212@gmail.com
- GitHub: [alvaroG-IA](https://github.com/alvaroG-IA)
- LinkedIn: [Álvaro García Velasco](https://www.linkedin.com/in/alvaro-garcia-velasco/)
