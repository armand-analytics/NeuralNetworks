# Proyecto de Clasificación de Dígitos MNIST

Este proyecto utiliza PyTorch para desarrollar un clasificador de dígitos escrito a mano usando el conjunto de datos MNIST. El siguiente diagrama de flujo muestra los pasos principales en el proceso de entrenamiento y validación del modelo.

## Diagrama de Flujo para el Entrenamiento y Validación del Modelo

```mermaid
graph TD
    A[Cargar Datos] --> B[Preprocesar Datos]
    B --> C[Entrenar Modelo]
    C --> D[Validar Modelo]
    D --> E[Evaluar Resultados]
