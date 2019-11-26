# face2face

Proyecto para las materias de "Aprendizaje (Máquina)" y "Reconocimiento de
patrones" de la FI, UNAM, semestre 2020-1

## Propuesta de proyecto

Implementar una red neuronal basada en la arquitectura pix2pix que sea capaz de
generar rostros con expresiones faciales distintas a las proporcionadas en el
conjunto de datos de entrenamiento, tomando como referencia la posición de otro
rostro.

A continuación se muestra un ejemplo de la aproximación que se desea lograr.

![./imgs/f2f_example.png](./imgs/f2f_example.png)

## Integrantes

- **Aprendizaje (Máquina)**
  - Aguilar Enriquez, Paul Sebastian
  - Cabrera López, Oscar Emilio
- **Reconocimiento de patrones**
  - Aguilar Enriquez, Paul Sebastian
  - Padilla Herrera Carlos Ignacio
  - Ramírez Ancona Simón Eduardo

## Requerimientos

- `tensorflow-gpu`
- `matplotlib`
- `numpy`
- `pydot`
- `pillow`
- `opencv`

## Referencias

- [Site: Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)
- [Generando FLORES realistas con IA - Pix2Pix | IA NOTEBOOK #5](https://www.youtube.com/watch?v=YsrMGcgfETY)
- [Pix2Pix con TensorFlow](https://www.tensorflow.org/tutorials/generative/pix2pix)
- [Edge to Artworks translation with Pix2Pix model. ](https://github.com/gallardorafael/edge2art)
- [Proyecto DeepFake que busca crear en ultima instancia caras falsas usando landmarks del controlador o entrada de texto ](https://github.com/RonyBenitez/mimix)
