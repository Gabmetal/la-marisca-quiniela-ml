# INFORME TÉCNICO EXPERIMENTAL: "Quiniela Chaos & ML Predictor"
**Proyecto:** La Marisca Project  
**Investigador / Cargo:** Autonomous Senior Data & ML Engineer  
**Fecha de Ejecución:** Viernes 16 de Octubre de 2026 (21:00 ART)  
**Objetivo Experimental:** Predicción Sábado 17 de Octubre de 2026  
**Loterías Auditadas:** Quiniela de la Ciudad (ex Nacional) y Quiniela de Formosa  

---

## 1. Resumen Ejecutivo y Marco Teórico
El presente proyecto implementa un pipeline experimental integral para evaluar el comportamiento estocástico de las quinielas oficiales argentinas (**Quiniela de la Ciudad** y **Quiniela de Formosa**) en su sorteo nocturno (21:00 ART), cubriendo el período comprendido entre el **1 de Julio de 2026 y el 16 de Octubre de 2026**.

El objetivo central es auditar la hipótesis nula de aleatoriedad equiprobable e independencia temporal ($H_0$), cuantificar el fenómeno de **sobreajuste espurio (overfitting)** que sufren los modelos de Machine Learning al entrenarse sobre generadores físicos de caos/ruido, y formular un portafolio de 5 recomendaciones analíticas para el sorteo del **Sábado 17 de Octubre de 2026**.

---

## 2. Auditoría Estadística e Información Empírica

### 2.1 Prueba de Uniformidad ($\chi^2$)
Para comprobar si los bolilleros presentan desgaste mecánico o sesgo de equiprobabilidad:
- **Ciudad:**
  - Decena: $\chi^2 = 6.033$ ($p = 0.7366 > 0.05$).
  - Unidad: $\chi^2 = 7.791$ ($p = 0.5553 > 0.05$).
  - Paridad: $\chi^2 = 0.011$ ($p = 0.9165 > 0.05$).
- **Formosa:**
  - Decena: $\chi^2 = 5.387$ ($p = 0.7993 > 0.05$).
  - Unidad: $\chi^2 = 9.688$ ($p = 0.3763 > 0.05$).
  - Paridad: $\chi^2 = 0.871$ ($p = 0.3507 > 0.05$).

**Conclusión Estadística:** En ambos bolilleros y en la muestra combinada ($N = 184$), **no se rechaza la hipótesis nula de uniformidad**. Las desviaciones observadas son perfectamente compatibles con fluctuaciones estocásticas de Poisson en muestras finitas.

### 2.2 Prueba de Rachas de Wald-Wolfowitz (Independencia Temporal)
Se evaluó si la secuencia de resultados exhibe memoria, inercia o alternancia no aleatoria:
- **Paridad Ciudad:** $Z = 1.477$, $p = 0.1396$ (Aleatorio).
- **Mediana Ciudad:** $Z = 0.212$, $p = 0.8321$ (Paseo aleatorio).
- **Paridad Formosa:** $Z = 0.513$, $p = 0.6081$ (Aleatorio).
- **Mediana Formosa:** $Z = -1.042$, $p = 0.2976$ (Aleatorio).

**Conclusión:** No existe correlación serial ni dependencia estocástica entre sorteos sucesivos. Cada extracción es estrictamente independiente.

### 2.3 Entropía de Shannon
La entropía observada frente al máximo teórico uniforme:
- **Entropía Decena:** $3.270$ bits en Ciudad ($98.43\%$ del máximo teórico $\log_2(10) = 3.3219$ bits).
- **Entropía Unidad:** $3.253$ bits en Ciudad ($97.94\%$ del máximo teórico).
- **Entropía Número 2D (Combinado):** $6.163$ bits ($92.76\%$ del máximo teórico $\log_2(100) = 6.6439$ bits).

La altísima densidad de información confirma que el generador físico produce ruido blanco casi puro.

---

## 3. Telemetría Ambiental y Coordenadas Astronómicas (17-Oct-2026)

Para el sorteo del **Sábado 17 de Octubre de 2026 a las 21:00 ART**, se integraron los vectores de telemetría física:

### Coordenadas y Pronóstico Meteorológico (Open-Meteo):
- **Ciudad Autónoma de Buenos Aires** (Lat: -34.6037, Lon: -58.3816):
  - Temperatura a 2m: **11.0 °C**
  - Humedad Relativa: **71 %**
  - Presión Superficial: **1020.6 hPa**
  - Velocidad del Viento (10m): **9.9 km/h**
- **Ciudad de Formosa** (Lat: -26.1849, Lon: -58.1731):
  - Temperatura a 2m: **16.9 °C**
  - Humedad Relativa: **81 %**
  - Presión Superficial: **1009.8 hPa**
  - Velocidad del Viento (10m): **9.6 km/h**

### Coordenadas Astronómicas y Fase Lunar:
- **Fase Sinódica:** 0.224 (Cuarto Creciente)
- **Iluminación Lunar:** **41.87%**
- **Transformada Cíclica:** $\sin(\theta) = 0.9867$, $\cos(\theta) = 0.1625$

---

## 4. Benchmark de Machine Learning: Sobreajuste vs Generalización

Para evitar la escasez muestral de un clasificador directo de 100 clases con $\approx 92$ observaciones, se implementó una arquitectura desacoplada de dos bolilleros independientes:
1. **Nivel 1:** Clasificadores especializados de Decena ($D \in [0..9]$) y Unidad ($U \in [0..9]$).
2. **Validación Temporal:** `TimeSeriesSplit(n_splits=5)` para medir generalización sin filtración de futuro (lookahead bias).
3. **Nivel 2:** Matriz de Probabilidad Conjunta $P(du) = P(D=d) \times P(U=u)$.

### Resultados del Benchmark (Ciudad):
| Modelo | Train Accuracy (%) | TimeSeriesSplit Test Acc (%) | Baseline Teórico (%) | Brecha de Sobreajuste (Overfit Gap) |
| :--- | :---: | :---: | :---: | :---: |
| **RandomForest** | 100.00% | 8.00% | 10.00% | +92.00% |
| **HistGradientBoosting** | 69.47% | 7.33% | 10.00% | +62.14% |
| **MLPClassifier** | 100.00% | 12.00% | 10.00% | +88.00% |

### Demostración Epistemológica:
Mientras los modelos en entrenamiento logran aprender correlaciones espurias sobre el ruido (100.0% en RandomForest), su desempeño en datos no vistos retrocede a la vecindad del azar puro ($\approx 10\%$). Esto demuestra rigurosamente que **las variables meteorológicas y lunares no alteran la física del bolillero de manera determinista**, sirviendo como un caso de estudio pedagógico clásico sobre el peligro de sobreajustar en series temporales caóticas.

---

## 5. Portafolio de Recomendaciones Predictivas (Top 5 Picks para la Rifa)

Con fines lúdicos y científicos ("for science and laughs"), presentamos el portafolio de 5 perfiles para el sorteo del **Sábado 17 de Octubre de 2026**:

### Quiniela de la Ciudad (ex Nacional)
| # | Perfil / Estrategia | Ambo (2D) | Decena | Unidad | Probabilidad del Modelo | Racionalidad y Fundamento |
| :-: | :--- | :-: | :-: | :-: | :-: | :--- |
| **1** | **ML Ensemble Top Joint** | **`23`** | 2 | 3 | 6.638% | Maximized joint P(D=d)*P(U=u) across RF, HistGB, MLP ensemble. |
| **2** | **Física & Luna Purista** | **`66`** | 6 | 6 | 5.428% | Weather forecast (11.0C, 1020.6hPa) + Lunar illumination (41.87%, Cuarto Creciente). |
| **3** | **Racha Caliente (Hot Streak)** | **`13`** | 1 | 3 | 6.004% | Statistical momentum: Decena 1 (13x) + Unidad 3 (13x) since July. |
| **4** | **Reversión a la Media (Atrasados)** | **`00`** | 0 | 0 | 0.054% | Longest drought: Decena 0 (36 draws) + Unidad 0 (28 draws). |
| **5** | **Monte Carlo Ponderado** | **`23`** | 2 | 3 | 6.638% | Stochastic draw sampled according to the normalized empirical posterior distribution. |

### Quiniela de Formosa
| # | Perfil / Estrategia | Ambo (2D) | Decena | Unidad | Probabilidad del Modelo | Racionalidad y Fundamento |
| :-: | :--- | :-: | :-: | :-: | :-: | :--- |
| **1** | **ML Ensemble Top Joint** | **`97`** | 9 | 7 | 8.702% | Maximized joint P(D=d)*P(U=u) across RF, HistGB, MLP ensemble. |
| **2** | **Física & Luna Purista** | **`07`** | 0 | 7 | 5.941% | Weather forecast (16.9C, 1009.8hPa) + Lunar illumination (41.87%, Cuarto Creciente). |
| **3** | **Racha Caliente (Hot Streak)** | **`79`** | 7 | 9 | 0.220% | Statistical momentum: Decena 7 (12x) + Unidad 9 (14x) since July. |
| **4** | **Reversión a la Media (Atrasados)** | **`76`** | 7 | 6 | 0.206% | Longest drought: Decena 7 (30 draws) + Unidad 6 (20 draws). |
| **5** | **Monte Carlo Ponderado** | **`40`** | 4 | 0 | 1.337% | Stochastic draw sampled according to the normalized empirical posterior distribution. |

---

## 6. Galería de Visualizaciones Analíticas

1. **Matriz de Probabilidad Conjunta 10x10:**  
   ![Heatmap Probabilidades 17 Octubre](file:///C:/Users/gabme/Documents/Repos/raffle/quiniela-ml-lab/src/../outputs/charts/heatmap_10x10_probabilidades_17oct.png)

2. **Benchmark de Sobreajuste (Overfitting Benchmark):**  
   ![Overfitting Benchmark](file:///C:/Users/gabme/Documents/Repos/raffle/quiniela-ml-lab/src/../outputs/charts/overfitting_benchmark.png)

3. **Importancia de Características Físicas vs Mecánicas:**  
   ![Feature Importance](file:///C:/Users/gabme/Documents/Repos/raffle/quiniela-ml-lab/src/../outputs/charts/feature_importance_fisica.png)

4. **Auditoría Empírica de Frecuencias y Atrasos:**  
   ![Frecuencias y Atrasos](file:///C:/Users/gabme/Documents/Repos/raffle/quiniela-ml-lab/src/../outputs/charts/frecuencias_y_atrasos.png)

---
*Reporte generado automáticamente por el pipeline experimental "Quiniela Chaos & ML Predictor".*
