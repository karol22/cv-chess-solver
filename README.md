# Chess Master

### *Computer Vision project using OpenCV*

by [Karol Gutierrez](https://github.com/karol22), [Guillermo Herrera](https://github.com/memoherreraacosta), [Juan Quirino](https://github.com/QuirinoC)

---

The objective of this project is to make a 1:1 chess game against a computer using [OpenCV for Python](https://docs.opencv.org/master/) where a camera will detect a chessboard and the game pieces from the upper view. The system will give instructions to the main user to move the pieces in order to keep the game.

The following diagram explains how the model is built using the *image recognition*, *Machine Learning* and *artifitial inteligence using the stockfish engine*:

![High level workflow](Data/diagram.png)

This process is divided in some steps:

1. Data mining/processing:
   - As a first step, it was used an image classificator to detect the pieces that are involved in the game, the dataset used was built from images taken from a camera, multiple images of each piece and it's player's label (red or blue for this specific case) and multiple backgrounds.
   The result where the following 14 labels `alfil_azul, caballo_azul, fondo_blanco, peon_azul, reina_azul, rey_azul, torre_azul, alfil_rojo, caballo_rojo, fondo_negro, peon_rojo, reina_rojo, rey_rojo, torre_rojo`.

   - In order to have a large dataset, we applied multiple transformations as Salt & Pepper filter, rotation, crop and resizing using [PyTorch](https://pytorch.org/) and [OpenCV](https://docs.opencv.org/master/). We collected aprox 32,000 different images as result of these transformations.

2. Data processing:
    - That dataset was processed in a Neuronal Network in order to get the label's characteristics in order that the system will classify each square board given, having this information, the board can be represented in a structure that can be easily processed to get the game's data.
  

3. Data Visualization:

**Important**

- [Python 3.8](https://www.python.org/downloads/release/python-380/) (at least) is required.
  
- The script is watching the chessboard in a fixed position, so it needs to be calibrated manually.

- The environment where the project ran was in a [google collab project](https://colab.research.google.com/notebooks/intro.ipynb) where it was indicated that needed to be set in an enviornment that uses a GPU.
