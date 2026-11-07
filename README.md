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
| `clientes_simulados_cobranza.csv` | 5.000 clientes simulados |
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
- **Variables:** días de atraso, relación cuota/ingreso, antigüedad y saldo promedio. Se excluyó `n_productos` por ser discreta (no compatible con el supuesto de normalidad multivariada).
- **Sin rebalanceo de clases** (por ejemplo, SMOTE), porque descalibra las probabilidades y empeora la decisión por costos.
- **Partición** 70/30 estratificada, semilla 42.

## Despliegue

Esta versión está publicada en Streamlit Community Cloud solo porque los datos son simulados. En producción real, el código estaría en un repositorio interno y la app correría en servidores propios de la institución (on-premises).
