# Proyecto de Clasificación de Dígitos MNIST

Este proyecto utiliza PyTorch para desarrollar un clasificador de dígitos escrito a mano usando el conjunto de datos MNIST. El siguiente diagrama de flujo muestra los pasos principales en el proceso de entrenamiento y validación del modelo.

## Diagrama de Flujo para el Entrenamiento y Validación del Modelo

```mermaid
graph TD
    subgraph Inicio
        A[Definir el modelo de red neuronal]
        B(Crear instancia de la red neuronal)
        C(Calcular número de parámetros)
    end

    subgraph Entrenamiento
        D[Configurar dispositivo]
        E(Definir función de pérdida y optimizador)
        F(Función de entrenamiento)
    end

    subgraph Validación
        G(Función de validación)
        H[Entrenamiento y validación del modelo]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H -->|Fin| Fin

