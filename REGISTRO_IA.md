# Registro de uso de IA, errores y correcciones

Cada fila es una interacción con la IA durante la construcción de la app. Este registro es la base de la memoria final: qué aportó la IA, qué errores aparecieron, qué se corrigió y qué decisiones requirieron criterio profesional.

**Columna "Tipo":** **A** = aporte de la IA · **E** = error o limitación de la IA · **C** = corrección propia · **P** = decisión de criterio profesional

## Etapa 1 — Versión mínima

| # | Fecha | Qué se pidió | Qué devolvió la IA | Problema o decisión | Quién lo detectó o decidió | Tipo |
|---|---|---|---|---|---|---|
| 1 | 26/10/2026 | Idea y estructura de la app para el curso | Propuesta de app en dos etapas (QDA + matriz de costos), pestañas y mapeo a las etapas del curso | — | — | A |
| 2 | 26/10/2026 | Especificación de la versión mínima | Borrador con 5 variables, LDA, 3 acciones genéricas más una de revisión manual, y escalado de variables | Las acciones y los costos no eran los de la simulación. Se reemplazaron por las 4 acciones reales con la matriz en pesos ya definida | Mati | C |
| 3 | 26/10/2026 | Ídem | La especificación proponía LDA | Se eligió **QDA**, según la validación de supuestos (covarianzas distintas por clase) | Mati | P |
| 4 | 26/10/2026 | Ídem | La especificación incluía `n_productos` | Se excluyó por ser discreta (de 1 a 8), incompatible con normalidad multivariada. **Revertida en la fila 12** | Mati | P |
| 5 | 26/10/2026 | Ídem | La especificación incluía escalar las variables antes del modelo | Innecesario: QDA es invariante a transformaciones afines y escalar no cambia las posteriores. Se eliminó | Detectado al pasar de la especificación al código; validado por Mati | E / C |
| 6 | 26/10/2026 | Código de la versión mínima | `app.py` funcional: posteriores, riesgo esperado, acción recomendada y matriz editable | — | — | A |
| 7 | 26/10/2026 | Prueba automática de la app | El código usaba un parámetro de Streamlit obsoleto (`use_container_width`) | Se reemplazó por la forma vigente (`width="stretch"`) | Prueba automática | E / C |
| 8 | 26/10/2026 | Ídem | Los montos se mostraban con formato inglés ($ 4,000) | Se cambió a formato argentino ($ 4.000) | Revisión de la salida | E / C |
| 9 | 07/11/2026 | Preparación del repositorio | `requirements.txt` sin versiones fijas | Se fijaron las versiones exactas para garantizar reproducibilidad | Revisión antes de publicar | C |
| 10 | 07/11/2026 | Validación de la app contra los notebooks de la investigación | La IA comparó la app con los notebooks 0b, 3, 5 y 07 | **La app no reproducía la investigación:** usaba el CSV crudo y 4 variables, mientras que el pipeline usa `clientes_simulados_normal.csv` (normal multivariada en escala logit) y 5 variables. Costo medio de Bayes en test: app $ 74.939, notebook $ 78.054. La especificación inicial de la IA se armó sin revisar los notebooks | IA, al validar | E |
| 11 | 07/11/2026 | Despliegue | Guía paso a paso para GitHub y Streamlit Community Cloud | Repositorio público y app publicada con Python 3.13. Sin secretos, porque la app no usa servicios externos | Mati | A |
| 12 | 07/11/2026 | Alinear la app con la investigación | Nueva versión: entrena con el dataset normal y 5 variables, y transforma a escala logit los valores que carga el usuario | Se eligió replicar la investigación en lugar de mantener una versión simplificada. `n_productos` vuelve a entrar porque en escala logit deja de ser discreta | Mati | P / C |
| 13 | 07/11/2026 | Verificación de la versión alineada | — | Regla ingenua, triviales e ideal coinciden exactamente con el notebook 07. Bayes: $ 117.073.750 contra $ 117.081.450 (0,007 %), porque el dataset se regeneró con el código de `0b` y otra versión de scipy. Se resuelve subiendo el CSV original de la investigación | IA y Mati | E / C |

## Etapa 2 — Comparación de políticas

| # | Fecha | Qué se pidió | Qué devolvió la IA | Problema o decisión | Quién lo detectó o decidió | Tipo |
|---|---|---|---|---|---|---|
| 14 | 07/11/2026 | Sumar la evaluación económica del notebook 07 | Pestaña "Comparación de políticas": Bayes contra ingenua, triviales e ideal, con ahorro y brecha por cliente; se recalcula al editar la matriz de costos | — | — | A |
| 15 | 07/11/2026 | Alcance de la etapa 2 | La IA proponía incluir intervalos de confianza sobre el costo (notebook 08) | Se dejó fuera: todavía no está profundizado ese tema en la investigación | Mati | P |

## Etapa 3 — Simplificación y evaluación de casos nuevos

| # | Fecha | Qué se pidió | Qué devolvió la IA | Problema o decisión | Quién lo detectó o decidió | Tipo |
|---|---|---|---|---|---|---|
| 16 | 07/11/2026 | Revisión de la app por el usuario | — | La app era difícil de leer y no se encontraba un caso que se derivara a recupero | Mati | E |
| 17 | 07/11/2026 | Facilitar la lectura | Perfiles de ejemplo (uno por acción), resultado destacado con color y explicación en una línea, probabilidades como barras, detalle del cálculo plegado | Se buscaron en el conjunto de test perfiles típicos de cada acción y se verificó que cada ejemplo produzca la acción que dice | IA | A |
| 18 | 07/11/2026 | Vista de políticas más ejecutiva | Pestaña "Resultados del modelo": una frase con el ahorro, tres indicadores, comparación de cuatro políticas y explicación de por qué ahorra; el detalle queda plegado | La "regla ingenua" de la investigación se presenta como "regla actual" para un público de negocio | IA; a validar por Mati | A / P |
| 19 | 07/11/2026 | Evaluar casos nuevos con el modelo entrenado | Pestaña "Evaluar varios clientes": plantilla CSV, validación de columnas y rangos, resultados descargables | Se agregó un aviso para no subir datos reales, porque la app está en la nube | IA; criterio de gobierno de datos | A / P |
| 20 | 07/11/2026 | Prueba automática | La primera versión del lote usaba `st.stop()` ante un error del archivo, lo que también cortaba las pestañas siguientes | Se reemplazó por una función que termina con `return` | IA, al revisar | E / C |

## Limitaciones conocidas (se completa a medida que avanza)

- Los datos son simulados: la distribución real de clientes puede ser distinta.
- La matriz de costos es una estimación; si los costos cambian, cambia la política óptima.
- La normalidad multivariada se cumple por construcción en escala logit (los datos se simularon así); con datos reales habría que validarla.
- La app no incluye la acción de revisión manual.
- La app no informa la incertidumbre del resultado (intervalos de confianza sobre el costo).
- Los casos nuevos fuera de los rangos de la simulación no se evalúan; el modelo no extrapola de forma confiable.
