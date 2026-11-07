# Decisor de Cobranza Temprana

Aplicación en Streamlit que recomienda qué acción tomar con un cliente en cobranza temprana, eligiendo la de **menor costo esperado en pesos**.

Trabajo final del curso de IA aplicada (UNSJ). Surge de la línea de investigación sobre integración de análisis discriminante y teoría de la decisión estadística del Laboratorio de Matemática Aplicada (IDEI, UNSJ).

> **Datos 100 % simulados.** No se usan datos reales de clientes. La app **recomienda**; la decisión final es del analista.

## Cómo funciona

La app separa dos etapas:

1. **Etapa estadística.** Un modelo de Análisis Discriminante Cuadrático (QDA) estima la probabilidad de cada desenlace para el cliente $\mathbf{x}$:
   - $\omega_0$: se regulariza solo
   - $\omega_1$: regulariza con gestión
   - $\omega_2$: mora grave

2. **Etapa decisoria.** Con una matriz de costos $C(a_i, \omega_k)$ en pesos, calcula el riesgo esperado de cada acción y elige la de menor riesgo:

$$R(a_i \mid \mathbf{x}) = \sum_{k=0}^{2} C(a_i,\omega_k)\,P(\omega_k \mid \mathbf{x}) \qquad a^* = \arg\min_{a_i} R(a_i \mid \mathbf{x})$$

Acciones posibles: no hacer nada, recordatorio, llamado, derivar a recupero. La matriz de costos se puede editar en pantalla.

La app tiene cuatro pestañas:
- **Evaluar un cliente:** se cargan los datos del cliente (o un perfil de ejemplo) y la app muestra la acción recomendada, su costo esperado y la probabilidad de cada desenlace.
- **Evaluar varios clientes:** se sube un CSV con casos nuevos (hay una plantilla para descargar); el modelo ya entrenado devuelve la acción recomendada para cada uno y los resultados se pueden descargar.
- **Resultados del modelo:** vista ejecutiva del ahorro frente a la regla actual, medido en clientes de prueba.
- **Cómo funciona:** la matemática del modelo y de la decisión.

## Rol de la IA

La IA (un asistente de IA generativa) se usó **solo para construir la app**, no durante su uso. Durante el uso, la recomendación sale de una regla matemática explícita, sin llamadas a servicios externos. Por eso la app no requiere claves ni credenciales.

Motivos:
- Las decisiones sobre clientes deben ser **auditables y reproducibles**.
- En un entorno bancario real, los datos de clientes **no pueden enviarse a servicios externos**.

El detalle de cada consulta a la IA, los errores encontrados y las correcciones está en [`REGISTRO_IA.md`](REGISTRO_IA.md).

## Archivos

| Archivo | Contenido |
|---|---|
| `app.py` | La aplicación |
| `clientes_simulados_normal.csv` | 5.000 clientes simulados en escala logit; **es el que usa el modelo** |
| `clientes_simulados_cobranza.csv` | Simulación original en unidades reales, de la que se deriva la anterior |
| `requirements.txt` | Librerías, con versiones fijas para reproducibilidad |
| `REGISTRO_IA.md` | Registro de uso de IA, errores y correcciones |

## Cómo correrla en una computadora

Requiere Python 3.11 o superior.

```
pip install -r requirements.txt
streamlit run app.py
```

## Decisiones de diseño

- **QDA** en lugar de LDA, según la validación de supuestos (matrices de covarianza distintas por clase).
- **Datos en escala logit:** cada variable se lleva a (0, 1) con límites de negocio y se le aplica logit; luego se simula una normal multivariada por clase. Así el supuesto de normalidad de QDA es razonable. La app recibe valores en unidades reales y aplica la misma transformación.
- **Variables:** días de atraso, relación cuota/ingreso, antigüedad, saldo promedio y cantidad de productos.
- **Reproducibilidad:** el modelo replica el pipeline de la investigación (misma partición, mismo modelo, misma matriz de costos).
- **Sin rebalanceo de clases** (por ejemplo, SMOTE), porque descalibra las probabilidades y empeora la decisión por costos.
- **Partición** 70/30 estratificada, semilla 42.

## Despliegue

Esta versión está publicada en Streamlit Community Cloud solo porque los datos son simulados. En producción real, el código estaría en un repositorio interno y la app correría en servidores propios de la institución (on-premises).
