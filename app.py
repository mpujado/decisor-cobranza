"""
Decisor de Cobranza Temprana — Etapa 3 (simplificación y evaluación de casos nuevos)

Marco en dos etapas:
  1) Etapa estadística: QDA estima P(ω_k | x) para las 3 clases.
  2) Etapa decisoria:   a* = argmin_i  R(a_i | x) = Σ_k C(a_i, ω_k) · P(ω_k | x)

El modelo replica el pipeline de la investigación (notebooks 0b, 3, 5 y 07):
  - Datos: clientes_simulados_normal.csv (normal multivariada por clase en escala logit).
  - 5 variables, partición 70/30 estratificada, semilla 42, QDA sin rebalanceo.
El usuario carga valores en unidades reales y la app aplica la misma transformación logit.

Datos 100 % simulados. La app recomienda; la decisión final es del analista.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import logit
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración (idéntica a la investigación)
# ---------------------------------------------------------------------------
RUTA_CSV = Path(__file__).parent / "clientes_simulados_normal.csv"
SEMILLA = 42
PROPORCION_TEST = 0.30

VARIABLES = ["dias_atraso", "cuota_ingreso", "antiguedad_meses",
             "saldo_promedio_miles", "n_productos"]

# Límites de negocio usados en 0b para llevar cada variable a (0, 1) y luego a logit
LIMITES = {
    "dias_atraso": (0.5, 30.5),
    "cuota_ingreso": (0.0, 1.0),
    "antiguedad_meses": (0.0, 180.0),
    "saldo_promedio_miles": (0.0, 3000.0),
    "n_productos": (0.5, 8.5),
}

ETIQUETAS_VAR = {
    "dias_atraso": "Días de atraso",
    "cuota_ingreso": "Relación cuota / ingreso",
    "antiguedad_meses": "Antigüedad (meses)",
    "saldo_promedio_miles": "Saldo promedio (miles de $)",
    "n_productos": "Cantidad de productos",
}

# Rango válido que acepta la app para cada variable (estrictamente dentro de LIMITES)
RANGOS = {
    "dias_atraso": (1, 30),
    "cuota_ingreso": (0.01, 0.99),
    "antiguedad_meses": (1, 179),
    "saldo_promedio_miles": (1, 2999),
    "n_productos": (1, 8),
}

CLASES = ["Se regulariza solo", "Regulariza con gestión", "Mora grave"]
ACCIONES = ["No hacer nada", "Recordatorio", "Llamado", "Derivar a recupero"]

# Matriz de costos C(acción, clase) en pesos por cliente (notebook 6a / 07)
COSTOS_BASE = pd.DataFrame(
    [
        [0, 236_500, 765_000],
        [150, 142_050, 765_150],
        [4_000, 4_000, 769_000],
        [80_000, 80_000, 595_000],
    ],
    index=ACCIONES,
    columns=CLASES,
)

# Regla actual de referencia: clase más probable → acción asociada (notebook 07)
MAPEO_INGENUA = {0: 0, 1: 2, 2: 3}  # 0 → nada, 1 → llamado, 2 → derivar

# Perfiles de ejemplo: uno por cada acción que recomienda el modelo con la matriz base
EJEMPLOS = {
    "No hacer nada": dict(dias_atraso=1, cuota_ingreso=0.20, antiguedad_meses=90,
                          saldo_promedio_miles=1500, n_productos=5),
    "Recordatorio": dict(dias_atraso=3, cuota_ingreso=0.18, antiguedad_meses=70,
                         saldo_promedio_miles=1100, n_productos=4),
    "Llamado": dict(dias_atraso=12, cuota_ingreso=0.26, antiguedad_meses=45,
                    saldo_promedio_miles=600, n_productos=3),
    "Derivar a recupero": dict(dias_atraso=24, cuota_ingreso=0.42, antiguedad_meses=15,
                               saldo_promedio_miles=150, n_productos=2),
}

# Cómo se muestra cada acción recomendada
ESTILO_ACCION = {
    "No hacer nada": (st.success, "Es muy probable que el cliente se regularice solo: "
                      "cualquier gestión cuesta más de lo que evita."),
    "Recordatorio": (st.info, "El riesgo es bajo, pero no nulo: un recordatorio de $ 150 "
                     "es una protección barata."),
    "Llamado": (st.warning, "Hay una probabilidad relevante de que necesite gestión: "
                "un llamado de $ 4.000 evita un costo esperado mayor."),
    "Derivar a recupero": (st.error, "La probabilidad de mora grave es alta: derivar ahora "
                           "cuesta menos, en promedio, que esperar."),
}


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------
def a_logit(datos: pd.DataFrame) -> pd.DataFrame:
    """Valores en unidades reales → escala logit (la del modelo)."""
    return pd.DataFrame({v: logit((datos[v].astype(float) - a) / (b - a))
                         for v, (a, b) in LIMITES.items()})


def pesos(v: float) -> str:
    """Formato argentino: $ 1.234.567"""
    return "$ " + f"{v:,.0f}".replace(",", ".")


@st.cache_data
def cargar_datos(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta)


@st.cache_resource
def ajustar_modelo(df: pd.DataFrame):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[VARIABLES], df["clase"],
        test_size=PROPORCION_TEST, stratify=df["clase"], random_state=SEMILLA,
    )
    # Priors = proporciones de entrenamiento. Sin SMOTE ni rebalanceo.
    qda = QuadraticDiscriminantAnalysis().fit(X_tr, y_tr)
    return qda, X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy()


def evaluar(datos_reales: pd.DataFrame, C: np.ndarray):
    """Devuelve posteriores (n, 3), riesgos (n, 4) y acción óptima (n,)."""
    post = qda.predict_proba(a_logit(datos_reales))
    riesgos = post @ C.T
    return post, riesgos, riesgos.argmin(axis=1)


def evaluar_lote(archivo):
    """Valida el CSV subido y muestra la acción recomendada para cada cliente.
    Usa return (no st.stop) para no cortar el resto de las pestañas."""
    try:
        lote = pd.read_csv(archivo)
    except Exception:
        st.error("No se pudo leer el archivo. Verificá que sea un CSV separado por comas.")
        return

    faltantes = [v for v in VARIABLES if v not in lote.columns]
    if faltantes:
        st.error(f"Faltan columnas: {', '.join(faltantes)}. Usá la plantilla.")
        return

    numericos = lote[VARIABLES].apply(pd.to_numeric, errors="coerce")
    validos = numericos.notna().all(axis=1)
    for v, (lo, hi) in RANGOS.items():
        validos &= numericos[v].between(lo, hi)
    if (~validos).any():
        st.warning(f"{int((~validos).sum())} fila(s) con valores vacíos o fuera de rango "
                   "quedaron afuera de la evaluación.")
    if not validos.any():
        return

    post_l, riesgos_l, acc_l = evaluar(numericos[validos], C)
    resultado = lote[validos].copy()
    resultado["accion_recomendada"] = [ACCIONES[a] for a in acc_l]
    resultado["costo_esperado"] = riesgos_l.min(axis=1).round(0)
    for k, col in enumerate(["prob_se_regulariza_solo", "prob_regulariza_con_gestion",
                             "prob_mora_grave"]):
        resultado[col] = post_l[:, k].round(3)

    st.markdown(f"**{len(resultado)} clientes evaluados**")
    conteo = resultado["accion_recomendada"].value_counts().reindex(ACCIONES, fill_value=0)
    for col, (nombre, cant) in zip(st.columns(4), conteo.items()):
        col.metric(nombre, int(cant))
    st.dataframe(resultado, width="stretch", hide_index=True)
    st.download_button("Descargar resultados", resultado.to_csv(index=False),
                       file_name="resultados_evaluacion.csv", mime="text/csv")


def cargar_ejemplo(nombre: str):
    for v, valor in EJEMPLOS[nombre].items():
        st.session_state[f"in_{v}"] = valor


# ---------------------------------------------------------------------------
# Inicio
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Decisor de Cobranza Temprana", layout="wide")
st.title("Decisor de Cobranza Temprana")
st.caption("Recomienda qué hacer con un cliente con atraso reciente, eligiendo la acción "
           "de menor costo esperado. Datos simulados: la decisión final es del analista.")

if not RUTA_CSV.exists():
    st.error(f"No se encontró el archivo de datos: {RUTA_CSV.name}")
    st.stop()

df = cargar_datos(RUTA_CSV)
qda, X_tr, X_te, y_tr, y_te = ajustar_modelo(df)

with st.expander("Costos de cada acción (editable)"):
    st.write("Costo en pesos por cliente de cada acción, según cómo termine el cliente. "
             "Si cambiás un valor, todas las pestañas se recalculan.")
    costos = st.data_editor(COSTOS_BASE, width="stretch", key="costos")
if costos.isna().any().any() or (costos < 0).any().any():
    st.error("La matriz de costos debe tener todos los valores completos y no negativos.")
    st.stop()
C = costos.to_numpy(dtype=float)

for v, valor in EJEMPLOS["Llamado"].items():
    st.session_state.setdefault(f"in_{v}", valor)

tab_uno, tab_lote, tab_result, tab_metodo = st.tabs(
    ["Evaluar un cliente", "Evaluar varios clientes", "Resultados del modelo",
     "Cómo funciona"]
)

# =============================================================================
# Pestaña 1: evaluar un cliente
# =============================================================================
with tab_uno:
    st.markdown("**Probá con un perfil de ejemplo o cargá los datos de tu caso.**")
    cols = st.columns(len(EJEMPLOS))
    for col, nombre in zip(cols, EJEMPLOS):
        col.button(f"Ejemplo: {nombre.lower()}", on_click=cargar_ejemplo, args=(nombre,),
                   width="stretch")

    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        cliente = {
            "dias_atraso": st.number_input(ETIQUETAS_VAR["dias_atraso"], 1, 30, step=1,
                                           key="in_dias_atraso"),
            "cuota_ingreso": st.number_input(ETIQUETAS_VAR["cuota_ingreso"], 0.01, 0.99,
                                             step=0.01, format="%.2f", key="in_cuota_ingreso"),
            "antiguedad_meses": st.number_input(ETIQUETAS_VAR["antiguedad_meses"], 1, 179,
                                                step=1, key="in_antiguedad_meses"),
            "saldo_promedio_miles": st.number_input(ETIQUETAS_VAR["saldo_promedio_miles"],
                                                    1, 2999, step=10,
                                                    key="in_saldo_promedio_miles"),
            "n_productos": st.number_input(ETIQUETAS_VAR["n_productos"], 1, 8, step=1,
                                           key="in_n_productos"),
        }

    post, riesgos, acc = evaluar(pd.DataFrame([cliente]), C)
    post, riesgos, i_opt = post[0], riesgos[0], int(acc[0])
    accion = ACCIONES[i_opt]

    with col_out:
        aviso, explicacion = ESTILO_ACCION[accion]
        aviso(f"### {accion}\n{explicacion}")
        st.metric("Costo esperado de esta acción", pesos(riesgos[i_opt]))

        st.markdown("**¿Cómo es probable que termine este cliente?**")
        for nombre, p in zip(CLASES, post):
            st.progress(float(p), text=f"{nombre}: {p:.0%}")

        with st.expander("Ver el detalle del cálculo"):
            st.write("Costo esperado de cada acción: suma de los costos de la matriz, "
                     "ponderados por la probabilidad de cada desenlace.")
            st.dataframe(pd.DataFrame({"Costo esperado": riesgos}, index=ACCIONES)
                         .style.format(pesos).highlight_min(color="#c8e6c9"),
                         width="stretch")
            st.dataframe(pd.DataFrame({"Probabilidad": post}, index=CLASES)
                         .style.format("{:.1%}"), width="stretch")

# =============================================================================
# Pestaña 2: evaluar varios clientes (archivo CSV)
# =============================================================================
with tab_lote:
    st.markdown("Subí un archivo CSV con los clientes a evaluar. El modelo ya entrenado "
                "calcula la acción recomendada para cada uno.")
    st.warning("La app está publicada en la nube: **no subas datos reales de clientes**. "
               "Usá solo casos ficticios o de prueba.")

    plantilla = pd.DataFrame([{"id_cliente": f"C{i + 1}", **valores}
                              for i, valores in enumerate(EJEMPLOS.values())])
    st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False),
                       file_name="plantilla_clientes.csv", mime="text/csv")
    with st.expander("Formato del archivo"):
        st.write("Una fila por cliente. Columnas obligatorias (la columna `id_cliente` "
                 "es opcional):")
        st.dataframe(pd.DataFrame(
            {"Descripción": [ETIQUETAS_VAR[v] for v in VARIABLES],
             "Mínimo": [RANGOS[v][0] for v in VARIABLES],
             "Máximo": [RANGOS[v][1] for v in VARIABLES]}, index=VARIABLES),
            width="stretch")

    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        evaluar_lote(archivo)

# =============================================================================
# Pestaña 3: resultados del modelo (vista ejecutiva)
# =============================================================================
with tab_result:
    prob_te = qda.predict_proba(X_te)
    acc_app = (prob_te @ C.T).argmin(axis=1)
    acc_actual = np.vectorize(MAPEO_INGENUA.get)(prob_te.argmax(1))
    costo_app = C[acc_app, y_te].mean()
    costo_actual = C[acc_actual, y_te].mean()
    costo_llamar = C[np.full(len(y_te), 2), y_te].mean()
    costo_ideal = C[:, y_te].min(axis=0).mean()
    ahorro = costo_actual - costo_app

    st.markdown(f"#### Con la app, cada cliente cuesta en promedio {pesos(ahorro)} menos "
                f"({ahorro / costo_actual:.0%})")
    st.caption(f"Evaluado sobre {len(y_te):,} clientes de prueba que el modelo no vio "
               "durante el entrenamiento.".replace(",", "."))

    m1, m2, m3 = st.columns(3)
    m1.metric("Costo medio con la app", pesos(costo_app))
    m2.metric("Costo medio con la regla actual", pesos(costo_actual),
              help="Regla actual: se asume que el cliente termina en su desenlace más "
                   "probable y se aplica la acción asociada (nada, llamado o derivación).")
    m3.metric("Ahorro por cliente", pesos(ahorro), f"{ahorro / costo_actual:.1%}"
              .replace(".", ","))

    st.markdown("**Costo medio por cliente según la política**")
    comparacion = [("Regla actual", costo_actual), ("Llamar a todos", costo_llamar),
                   ("Con la app", costo_app),
                   ("Ideal (si se supiera de antemano cómo termina cada cliente)", costo_ideal)]
    for nombre, valor in comparacion:
        st.progress(float(valor / costo_actual), text=f"{nombre}: {pesos(valor)}")

    st.markdown("**Por qué ahorra:** la regla actual no hace nada con los clientes que "
                "probablemente se regularicen solos, pero una parte de ellos sí necesitaba "
                "gestión. La app considera también ese riesgo y, como un llamado cuesta poco "
                "frente a una mora, decide llamar en más casos.")

    with st.expander("Ver el detalle"):
        n = len(y_te)
        politicas = {
            "Con la app (regla de Bayes)": acc_app,
            "Regla actual (desenlace más probable)": acc_actual,
            "No hacer nada a nadie": np.full(n, 0),
            "Recordatorio a todos": np.full(n, 1),
            "Llamar a todos": np.full(n, 2),
            "Ideal (si se supiera el desenlace)": C[:, y_te].argmin(axis=0),
        }
        filas = []
        for nombre, acc in politicas.items():
            costo = C[acc, y_te]
            filas.append({"Política": nombre, "Costo total": pesos(costo.sum()),
                          "Costo medio por cliente": pesos(costo.mean()),
                          **{a: int((acc == i).sum()) for i, a in enumerate(ACCIONES)}})
        st.dataframe(pd.DataFrame(filas).set_index("Política"), width="stretch")

# =============================================================================
# Pestaña 4: cómo funciona
# =============================================================================
with tab_metodo:
    st.markdown("**1. Etapa estadística (QDA)**")
    st.write("Cada desenlace se modela como una normal multivariada con su propia media y "
             "su propia matriz de covarianzas. Con las probabilidades a priori π_k "
             "(proporciones de entrenamiento) se obtiene la probabilidad de cada desenlace:")
    st.latex(r"P(\omega_k \mid \mathbf{x}) = \frac{\pi_k\, f_k(\mathbf{x})}"
             r"{\sum_{j} \pi_j\, f_j(\mathbf{x})}")
    st.markdown("**2. Etapa decisoria (regla de Bayes con costos)**")
    st.latex(r"R(a_i \mid \mathbf{x}) = \sum_{k=0}^{2} C(a_i,\omega_k)\,P(\omega_k \mid \mathbf{x})"
             r"\qquad a^* = \arg\min_{a_i} R(a_i \mid \mathbf{x})")
    st.write("La acción recomendada no tiene por qué coincidir con el desenlace más "
             "probable: la matriz de costos puede empujar la decisión hacia la acción que "
             "evita el error más caro.")
    st.markdown("**3. Escala de las variables**")
    st.write("Los datos simulados están en escala logit: cada variable se lleva a (0, 1) con "
             "límites de negocio y luego se aplica logit, para que el supuesto de normalidad "
             "multivariada de QDA sea razonable. La app recibe los valores en unidades reales "
             "y aplica la misma transformación:")
    st.latex(r"z = \operatorname{logit}\!\left(\frac{x - a}{b - a}\right)")
    st.dataframe(pd.DataFrame(LIMITES, index=["a (mínimo)", "b (máximo)"]).T
                 .rename(index=ETIQUETAS_VAR), width="stretch")
    st.caption(f"Entrenamiento: {len(y_tr):,} clientes · Test: {len(y_te):,} clientes · "
               f"Semilla: {SEMILLA}".replace(",", "."))
