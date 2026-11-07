# Registro de uso de IA, errores y correcciones

Cada fila es una interacción con la IA durante la construcción de la app. Este registro es la base de la memoria final: qué aportó la IA, qué errores aparecieron, qué se corrigió y qué decisiones requirieron criterio profesional.

**Columna "Tipo":** **A** = aporte de la IA · **E** = error o limitación de la IA · **C** = corrección propia · **P** = decisión de criterio profesional

## Etapa 1 — Versión mínima

| # | Fecha | Qué se pidió | Qué devolvió la IA | Problema o decisión | Quién lo detectó o decidió | Tipo |
|---|---|---|---|---|---|---|
| 1 | 26/10/2026 | Idea y estructura de la app para el curso | Propuesta de app en dos etapas (QDA + matriz de costos), pestañas y mapeo a las etapas del curso | — | — | A |
| 2 | 26/10/2026 | Especificación de la versión mínima | Borrador con 5 variables, LDA, 3 acciones genéricas más una de revisión manual, y escalado de variables | Las acciones y los costos no eran los de la simulación. Se reemplazaron por las 4 acciones reales con la matriz en pesos ya definida | Mati | C |
| 3 | 26/10/2026 | Ídem | La especificación proponía LDA | Se eligió **QDA**, según la validación de supuestos (covarianzas distintas por clase) | Mati | P |
| 4 | 26/10/2026 | Ídem | La especificación incluía `n_productos` | Se excluyó por ser discreta (de 1 a 8), incompatible con normalidad multivariada | Mati | P |
| 5 | 26/10/2026 | Ídem | La especificación incluía escalar las variables antes del modelo | Innecesario: QDA es invariante a transformaciones afines y escalar no cambia las posteriores. Se eliminó | Detectado al pasar de la especificación al código; validado por Mati | E / C |
| 6 | 26/10/2026 | Código de la versión mínima | `app.py` funcional: posteriores, riesgo esperado, acción recomendada y matriz editable | — | — | A |
| 7 | 26/10/2026 | Prueba automática de la app | El código usaba un parámetro de Streamlit obsoleto (`use_container_width`) | Se reemplazó por la forma vigente (`width="stretch"`) | Prueba automática | E / C |
| 8 | 26/10/2026 | Ídem | Los montos se mostraban con formato inglés ($ 4,000) | Se cambió a formato argentino ($ 4.000) | Revisión de la salida | E / C |
| 9 | 07/11/2026 | Preparación del repositorio | `requirements.txt` sin versiones fijas | Se fijaron las versiones exactas para garantizar reproducibilidad | Revisión antes de publicar | C |
| 10 | pendiente | Validación contra el notebook | — | Contrastar posteriores y acción de 3 clientes de control con `5 - Ajuste y evaluación.ipynb` | Mati | — |

## Limitaciones conocidas (se completa a medida que avanza)

- Los datos son simulados: la distribución real de clientes puede ser distinta.
- La matriz de costos es una estimación; si los costos cambian, cambia la política óptima.
- QDA supone normalidad multivariada dentro de cada clase; las variables acotadas (por ejemplo, días de atraso entre 0 y 30) la cumplen solo de forma aproximada.
- La versión mínima no incluye la acción de revisión manual.
