
# =================== Cargar Datos ===================
import time
import joblib
import os
import gdown
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# Cargar el dataset primero para poder usarlo en toda la aplicación
df = sns.load_dataset("diamonds")

# =================== Configuración de Página ===================
st.set_page_config(
    page_title="Diamond Analytics | Machine Learning",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== Sidebar de Navegación ===================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3039/3039513.png", width=80)
st.sidebar.title("💎 Diamond Analytics")
st.sidebar.markdown("---")

# Menú de navegación con íconos y diseño mejorado
st.sidebar.markdown("### 📌 Navegación")
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📊 Análisis Exploratorio", "📈 Regresión", "⚡ Clasificación", "🧪 Simulador"],
    format_func=lambda x: f"**{x}**"
)

# Información del dataset en sidebar
with st.sidebar.expander("ℹ️ Información del Dataset"):
    st.markdown(f"""
    - **Registros:** {df.shape[0]:,}
    - **Variables:** {df.shape[1]}
    - **Memoria:** {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado por: **Rafael Travado Fernández**")
st.sidebar.markdown("Bootcamp Data Science 2025")
st.sidebar.markdown("[GitHub](https://github.com/Rafael-TF) | [LinkedIn](https://www.linkedin.com/in/rafael-travado-4a1b6437/) | [Portfolio](https://rafaeltravado.netlify.app/)")

# =================== Página de Inicio ===================
if seccion == "🏠 Inicio":
    # Header con diseño atractivo
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h1 style='color: #3A86FF;'>💎 DIAMOND ANALYTICS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 20px; color: #555;'>Análisis avanzado y predicción con Machine Learning</p>", unsafe_allow_html=True)
    
    with col2:
        st.image("https://blascojoyero.com/wp-content/uploads/2020/09/diamante-050-blasco-joyero-GVS2-river..jpg", width=200)
    
    st.markdown("---")
    
    # Sección de resumen con KPIs clave
    st.subheader("📈 Resumen del Dataset")
    
    # Mostrar KPIs en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Precio Promedio",
            value=f"${df['price'].mean():,.2f}",
            delta=f"{(df['price'].mean() / df['price'].median() - 1) * 100:.1f}% vs mediana"
        )
    
    with col2:
        st.metric(
            label="Quilates Promedio",
            value=f"{df['carat'].mean():.2f}",
            delta=f"{df['carat'].max():.2f} máx"
        )
    
    with col3:
        corte_top = df['cut'].value_counts().idxmax()
        st.metric(
            label="Corte más común",
            value=f"{corte_top}",
            delta=f"{df['cut'].value_counts()[corte_top] / len(df) * 100:.1f}% del total"
        )
    
    with col4:
        color_top = df['color'].value_counts().idxmax()
        st.metric(
            label="Color predominante",
            value=f"{color_top}",
            delta=f"{df['color'].value_counts()[color_top] / len(df) * 100:.1f}% del total"
        )
    
    # Dashboard principal con tabs para mejor organización
    st.markdown("## 🔍 Descubre el mundo de los diamantes")
    
    tab1, tab2, tab3 = st.tabs(["🌟 Proyecto", "📊 Vista Previa", "🧠 Modelos"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### 🚀 Bienvenido a Diamond Analytics
            
            Este proyecto utiliza **ciencia de datos y aprendizaje automático** para analizar el conjunto de datos `diamonds` de seaborn, que contiene información detallada sobre **53,940 diamantes** con sus características y precios.
            
            💫 **¿Qué hace único a este análisis?**
            - Visualizaciones interactivas con filtros dinámicos
            - Modelos predictivos con métricas en tiempo real
            - Simulador para estimar precios personalizados
            - Insights del mercado basados en datos reales
            
            👨‍💻 **Tecnologías utilizadas:**
            - Python | Pandas | NumPy
            - Scikit-learn | Matplotlib | Seaborn
            - Streamlit | Plotly
            """)
            
            st.info("Navega por las diferentes secciones usando el menú lateral izquierdo para descubrir todos los análisis disponibles.")
        
        with col2:
            # Diagrama de radar mostrando características importantes
            # Datos para gráfico radial
            categories = ['Quilates', 'Profundidad', 'Tabla', 'Precio', 'Dimensión x', 'Dimensión y']
            
            # Normalizar valores para el gráfico radial
            values = [
                df['carat'].mean() / df['carat'].max(),
                df['depth'].mean() / df['depth'].max(),
                df['table'].mean() / df['table'].max(),
                df['price'].mean() / df['price'].max(),
                df['x'].mean() / df['x'].max(),
                df['y'].mean() / df['y'].max()
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                line_color='#3A86FF',
                fillcolor='rgba(58, 134, 255, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Características Promedio (Normalizado)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Vista previa de los datos")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Permitir filtrar por columnas principales
            cortes = st.multiselect("Filtrar por corte:", options=sorted(df['cut'].unique()), default=None)
            precio_rango = st.slider("Rango de precios ($):", min_value=int(df['price'].min()), max_value=int(df['price'].max()), 
                                    value=[int(df['price'].min()), int(df['price'].max())])
            
            # Aplicar filtros
            df_filtrado = df.copy()
            if cortes:
                df_filtrado = df_filtrado[df_filtrado['cut'].isin(cortes)]
            df_filtrado = df_filtrado[(df_filtrado['price'] >= precio_rango[0]) & (df_filtrado['price'] <= precio_rango[1])]
            
            # Mostrar conteo
            st.write(f"Mostrando {len(df_filtrado):,} de {len(df):,} diamantes")
        
        with col2:
            # Tabla con estilo mejorado
            st.dataframe(
                df_filtrado.head(10).style.background_gradient(cmap='Blues', subset=['price', 'carat']).format({
                    'price': '${:.2f}',
                    'carat': '{:.2f}',
                    'depth': '{:.1f}%',
                    'x': '{:.2f} mm',
                    'y': '{:.2f} mm',
                    'z': '{:.2f} mm'
                }),
                use_container_width=True
            )
            
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.markdown("#### 📋 Tipos de variables")
                st.write(f"**Numéricas:** {df.select_dtypes(include=['float64', 'int64']).columns.tolist()}")
                st.write(f"**Categóricas:** {df.select_dtypes(include=['object', 'category']).columns.tolist()}")
            
            with col_stats2:
                # Mostrar valores faltantes si los hay
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    st.markdown("#### 🚫 Valores faltantes")
                    st.write(missing[missing > 0])
                else:
                    st.markdown("#### ✅ Calidad de datos")
                    st.success("¡El dataset está completo! No hay valores faltantes.")
    
    with tab3:
        st.subheader("Modelos de Machine Learning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📈 Modelo de Regresión
            
            Predice el **precio de un diamante** basado en sus características físicas.
            
            **Variables utilizadas:**
            - Quilates (peso)
            - Dimensiones (x, y, z)
            - Claridad
            - Color
            - Profundidad y tabla
            
            **Técnicas aplicadas:**
            - Regresión Lineal
            - Random Forest
            - Gradient Boosting
            
            📌 *Navega a la sección de Regresión para ver el modelo completo*
            """)
        
        with col2:
            st.markdown("""
            ### ⚡ Modelo de Clasificación
            
            Predice la **calidad del corte** de un diamante basado en sus atributos.
            
            **Variables utilizadas:**
            - Quilates
            - Dimensiones
            - Precio
            - Proporciones
            
            **Técnicas aplicadas:**
            - Árboles de Decisión
            - SVM
            - Redes Neuronales
            
            📌 *Navega a la sección de Clasificación para ver el modelo completo*
            """)
    
    # Sección de recursos adicionales
    st.markdown("---")
    st.markdown("## 📚 Recursos y Referencias")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📗 Documentación
        - [Dataset Diamonds](https://ggplot2.tidyverse.org/reference/diamonds.html)
        - [Guía de diamantes GIA](https://4cs.gia.edu/)
        - [Streamlit Docs](https://docs.streamlit.io/)
        """)
    
    with col2:
        st.markdown("""
        ### 🧰 Herramientas
        - [Pandas](https://pandas.pydata.org/)
        - [Scikit-learn](https://scikit-learn.org/)
        - [Plotly](https://plotly.com/)
        - [Seaborn](https://seaborn.pydata.org/)
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Análisis complementarios
        - [Tendencias de precios de diamantes](https://www.diamonds.pro/education/diamond-prices/)
        - [Factors that Affect Diamond Prices](https://www.gemsociety.org/article/diamond-pricing-analysis/)
        """)
    
    # Call to action
    st.markdown("---")
    st.success("👈 Selecciona **'📊 Análisis Exploratorio'** en el menú de navegación para comenzar a explorar los datos!")
    
    # Footer con estilo
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p style='color: #555;'>Desarrollado para el Bootcamp de Data Science 2025</p>
        <p style='color: #777; font-size: 12px;'>Última actualización: Febrero 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
# =================== Análisis Exploratorio ===================
elif seccion == "📊 Análisis Exploratorio":
    st.title("📊 Análisis Exploratorio de Datos")
    
    # Banner estilizado con gradiente
    st.markdown("""
    <div style="background: linear-gradient(to right, #4364f7, #6fb1fc); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">🔍 Explorando los Factores que Afectan los Diamantes</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Descripción mejorada
    st.markdown("""
    En esta sección analizaremos de forma visual los factores que impactan en el **precio** (`price`) 
    y la **calidad del corte** (`cut`) de los diamantes. Los gráficos interactivos te permitirán
    descubrir patrones clave en los datos y comprender mejor las relaciones entre las variables.
    
    Utiliza los filtros en el panel lateral para personalizar tu análisis 👉
    """)
    
    # Contenedor con borde para los filtros
    st.sidebar.markdown("""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 10px; border-left: 5px solid #4364f7;">
        <h3 style="color: #4364f7; margin-top: 0;">🎛️ Filtros de Análisis</h3>
        <p style="font-size: 0.9em; color: #666;">Personaliza la visualización según tus intereses</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formulario de filtros con diseño mejorado
    with st.form("filters_form"):
        # Mejora visual de los controles con emojis y colores
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p style="color: #4364f7; font-weight: bold;">💎 Características</p>', unsafe_allow_html=True)
            cut_filter = st.multiselect("Tipo de Corte", df["cut"].unique(), default=df["cut"].unique())
            clarity_filter = st.multiselect("Claridad", df["clarity"].unique(), default=df["clarity"].unique())
        
        with col2:
            st.markdown('<p style="color: #4364f7; font-weight: bold;">🔍 Propiedades</p>', unsafe_allow_html=True)
            color_filter = st.multiselect("Color", df["color"].unique(), default=df["color"].unique())
            carat_range = st.slider("Rango de Quilates", 
                                   float(df["carat"].min()), 
                                   float(df["carat"].max()), 
                                   (float(df["carat"].min()), float(df["carat"].max())))
        
        # Botón con estilo
        submitted = st.form_submit_button("Aplicar Filtros", 
                                         use_container_width=True)
    
    # Filtrado de Datos
    df_filtered = df[
        (df["cut"].isin(cut_filter)) &
        (df["color"].isin(color_filter)) &
        (df["clarity"].isin(clarity_filter)) &
        (df["carat"] >= carat_range[0]) & (df["carat"] <= carat_range[1])
    ]
    
    # Métricas destacadas
    st.markdown("### 📌 Resumen de Datos Filtrados")
    
    # Mostrar métricas clave en forma de tarjetas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total de Diamantes", 
                 value=f"{len(df_filtered):,}", 
                 delta=f"{len(df_filtered)/len(df)*100:.1f}% del total")
    with col2:
        st.metric(label="Precio Promedio", 
                 value=f"${df_filtered['price'].mean():,.2f}")
    with col3:
        st.metric(label="Quilates Promedio", 
                 value=f"{df_filtered['carat'].mean():.2f}")
    
    # Dataframe más compacto sin estilo para evitar el error de límite de celdas
    with st.expander("Ver datos filtrados", expanded=False):
        st.dataframe(df_filtered, use_container_width=True, height=250)
    
    # Separador estilizado
    st.markdown("""<hr style="height:2px;border:none;color:#4364f7;background-color:#4364f7;margin:25px 0;" />""", 
               unsafe_allow_html=True)
    
    # Distribución de precios con tema coherente
    st.subheader("💰 Distribución de Precios")
    
    # Tabs para ofrecer diferentes visualizaciones
    precio_tab1, precio_tab2 = st.tabs(["Histograma (Plotly)", "Histograma (Seaborn)"])
    
    with precio_tab1:
        fig = px.histogram(df_filtered, x="price", 
                          color_discrete_sequence=["#4364f7"],
                          marginal="box",
                          nbins=30,
                          opacity=0.7,
                          title="Distribución de precios de los diamantes")
        fig.update_layout(
            plot_bgcolor="white",
            xaxis_title="Precio ($)",
            yaxis_title="Cantidad de diamantes",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with precio_tab2:
        # Gráfica de Seaborn - Histograma estilizado
        st.markdown('<p style="color: #4364f7; font-weight: bold; font-size: 14px;">Gráfico generado con Seaborn</p>', unsafe_allow_html=True)
        
        # Configuración de estilo de seaborn para mantener consistencia con el diseño
        sns.set_style("whitegrid")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histograma con KDE
        sns.histplot(df_filtered["price"], bins=30, kde=True, color="#4364f7", alpha=0.7, ax=ax)
        
        # Personalización para que combine con el diseño
        ax.set_title("Distribución del Precio de los Diamantes", fontsize=16, pad=20)
        ax.set_xlabel("Precio ($)", fontsize=12)
        ax.set_ylabel("Frecuencia", fontsize=12)
        
        # Mejorar apariencia general
        plt.tight_layout()
        
        # Mostrar el gráfico
        st.pyplot(fig)
    
    # Relación entre quilates y precio con mejor presentación
    st.markdown("""<div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h3 style="margin-top: 0;">📈 Relación entre Quilates y Precio</h3>
                </div>""", 
                unsafe_allow_html=True)
    
    scatter_tab1, scatter_tab2 = st.tabs(["Plotly Interactivo", "Seaborn"])
    
    with scatter_tab1:
        color_var = st.radio("Colorear por:", ["cut", "color", "clarity"], horizontal=True)
        
        fig = px.scatter(df_filtered, 
                        x="carat", 
                        y="price", 
                        color=color_var,
                        size="depth",
                        hover_name="cut",
                        hover_data=["clarity", "color", "table"],
                        title="Relación entre quilates y precio",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        opacity=0.7,
                        trendline="ols")
        
        fig.update_layout(
            plot_bgcolor="white",
            xaxis_title="Quilates",
            yaxis_title="Precio ($)",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with scatter_tab2:
        st.markdown('<p style="color: #4364f7; font-weight: bold; font-size: 14px;">Gráfico generado con Seaborn</p>', unsafe_allow_html=True)
        
        # Configurar un estilo limpio para Seaborn
        sns.set_style("whitegrid")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Si hay demasiados puntos, tomamos una muestra para no sobrecargar la visualización
        sample_size = min(5000, len(df_filtered))
        sampled_data = df_filtered.sample(sample_size, random_state=42) if len(df_filtered) > sample_size else df_filtered
        
        # Gráfico de dispersión con regresión
        sns.regplot(
            x="carat", 
            y="price", 
            data=sampled_data, 
            scatter_kws={"alpha": 0.6, "color": "#4364f7"}, 
            line_kws={"color": "#ff6b6b"},
            ax=ax
        )
        
        # Overlay con un scatterplot coloreado por tipo de corte
        sns.scatterplot(
            x="carat", 
            y="price", 
            hue="cut", 
            data=sampled_data, 
            alpha=0.6, 
            palette="coolwarm",
            ax=ax
        )
        
        # Personalización visual
        ax.set_title("Relación entre Quilates y Precio por Tipo de Corte", fontsize=16, pad=20)
        ax.set_xlabel("Quilates", fontsize=12)
        ax.set_ylabel("Precio ($)", fontsize=12)
        
        # Ajustar leyenda
        plt.legend(title="Tipo de Corte", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Mostrar el gráfico
        st.pyplot(fig)
    
    # Comparativa de precios por tipo de corte
    st.markdown("""<div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h3 style="margin-top: 0;">📊 Análisis Comparativo</h3>
                </div>""", 
                unsafe_allow_html=True)
    
    comparativa_tab1, comparativa_tab2 = st.tabs(["Plotly", "Seaborn"])
    
    with comparativa_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df_filtered, 
                        x="cut", 
                        y="price", 
                        color="cut",
                        title="Precio según tipo de corte",
                        color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(
                plot_bgcolor="white",
                xaxis_title="Tipo de corte",
                yaxis_title="Precio ($)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df_filtered, 
                        x="color", 
                        y="price", 
                        color="color",
                        title="Precio según color",
                        color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(
                plot_bgcolor="white",
                xaxis_title="Color",
                yaxis_title="Precio ($)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with comparativa_tab2:
        st.markdown('<p style="color: #4364f7; font-weight: bold; font-size: 14px;">Gráficos generados con Seaborn</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot con Seaborn para Cut
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Personalizar la apariencia de seaborn
            sns.set_style("whitegrid")
            sns.set_palette("coolwarm")
            
            # Crear boxplot
            sns.boxplot(x="cut", y="price", data=df_filtered, ax=ax, palette="coolwarm")
            
            # Añadir swarmplot con puntos para mejor visualización de la distribución
            sns.swarmplot(x="cut", y="price", data=df_filtered.sample(min(500, len(df_filtered))), 
                          color="black", alpha=0.5, ax=ax, size=3)
            
            # Personalización visual
            ax.set_title("Precio según Tipo de Corte", fontsize=14, pad=20)
            ax.set_xlabel("Tipo de Corte", fontsize=12)
            ax.set_ylabel("Precio ($)", fontsize=12)
            plt.xticks(rotation=45)
            
            # Ajustar diseño
            plt.tight_layout()
            
            # Mostrar el gráfico
            st.pyplot(fig)
            
        with col2:
            # Violinplot con Seaborn para Clarity
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Crear violinplot
            sns.violinplot(x="clarity", y="price", data=df_filtered, ax=ax, palette="viridis")
            
            # Personalización visual
            ax.set_title("Distribución de Precios por Claridad", fontsize=14, pad=20)
            ax.set_xlabel("Claridad", fontsize=12)
            ax.set_ylabel("Precio ($)", fontsize=12)
            plt.xticks(rotation=45)
            
            # Ajustar diseño
            plt.tight_layout()
            
            # Mostrar el gráfico
            st.pyplot(fig)
    
    # Matriz de correlación mejorada
    st.markdown("""<div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h3 style="margin-top: 0;">📉 Correlaciones entre Variables</h3>
                </div>""", 
                unsafe_allow_html=True)
    
    corr_tab1, corr_tab2 = st.tabs(["Plotly", "Seaborn"])
    
    with corr_tab1:
        # Matriz con heatmap interactivo de Plotly
        corr_matrix = df_filtered.select_dtypes(include=['number']).corr()
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       color_continuous_scale='RdBu_r',
                       title="Matriz de Correlación entre Variables Numéricas",
                       aspect="auto")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with corr_tab2:
        st.markdown('<p style="color: #4364f7; font-weight: bold; font-size: 14px;">Gráfico generado con Seaborn</p>', unsafe_allow_html=True)
        
        # Heatmap de correlación con Seaborn
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calcular correlación
        corr_matrix = df_filtered.select_dtypes(include=['number']).corr()
        
        # Crear máscara para triángulo superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Crear heatmap con máscara
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm', 
                   mask=mask, 
                   linewidths=0.5, 
                   cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        # Personalización
        ax.set_title("Matriz de Correlación (Triángulo Inferior)", fontsize=16, pad=20)
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Mostrar el gráfico
        st.pyplot(fig)
    
    # Añadir gráfico de pares con Seaborn
    st.markdown("""<div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h3 style="margin-top: 0;">🔄 Relaciones Multivariables (Seaborn)</h3>
                </div>""", 
                unsafe_allow_html=True)
    
    # Seleccionar variables para el pairplot
    st.markdown('<p style="color: #4364f7; font-weight: bold; font-size: 14px;">Selecciona variables para el análisis de pares</p>', unsafe_allow_html=True)
    pairplot_vars = st.multiselect(
        "Variables a incluir:", 
        options=['carat', 'depth', 'table', 'price', 'x', 'y', 'z'],
        default=['carat', 'price', 'depth']
    )
    
    if len(pairplot_vars) >= 2:
        # Mostrar advertencia si hay muchos datos
        if len(df_filtered) > 1000:
            st.warning("⚠️ Se tomará una muestra de 1000 diamantes para generar el pairplot y mantener un buen rendimiento.")
            df_sample = df_filtered.sample(1000, random_state=42)
        else:
            df_sample = df_filtered
        
        # Color del pairplot
        color_var = st.radio("Variable de color para el pairplot:", ["cut", "clarity", "color"], horizontal=True)
        
        # Crear pairplot
        fig = plt.figure(figsize=(12, 10))
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # Generar pairplot
        g = sns.pairplot(
            df_sample, 
            vars=pairplot_vars, 
            hue=color_var, 
            palette="coolwarm", 
            diag_kind="kde",
            plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
            diag_kws={'alpha': 0.6, 'fill': True}
        )
        
        # Personalizar título
        g.fig.suptitle(f"Relaciones entre variables seleccionadas (coloreado por {color_var})", 
                      fontsize=16, y=1.02)
        
        # Mostrar pairplot
        st.pyplot(g.fig)
    else:
        st.info("ℹ️ Selecciona al menos 2 variables para generar el pairplot.")
    
    # Análisis adicional con KDE bivariado
    st.markdown("""<div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h3 style="margin-top: 0;">🌊 Densidad Bivariada (Seaborn)</h3>
                </div>""", 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Variable X para densidad:", ["carat", "depth", "table", "price", "x", "y", "z"], index=0)
    with col2:
        y_var = st.selectbox("Variable Y para densidad:", ["carat", "depth", "table", "price", "x", "y", "z"], index=3)
    
    if x_var != y_var:
        # Configuración del gráfico
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # Si hay muchos puntos, tomar muestra
        if len(df_filtered) > 2000:
            df_sample = df_filtered.sample(2000, random_state=42)
        else:
            df_sample = df_filtered
        
        # Crear KDE bivariado
        sns.kdeplot(
            data=df_sample,
            x=x_var,
            y=y_var,
            fill=True,
            cmap="Blues",
            thresh=0.05,
            alpha=0.7,
            ax=ax
        )
        
        # Superponer scatterplot
        sns.scatterplot(
            data=df_sample,
            x=x_var,
            y=y_var,
            hue="cut",
            palette="coolwarm",
            alpha=0.6,
            ax=ax
        )
        
        # Personalización
        ax.set_title(f"Densidad Bivariada: {x_var} vs {y_var}", fontsize=16, pad=20)
        ax.set_xlabel(x_var, fontsize=12)
        ax.set_ylabel(y_var, fontsize=12)
        
        # Ajustar leyenda
        plt.legend(title="Tipo de Corte", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Mostrar el gráfico
        st.pyplot(fig)
    else:
        st.info("ℹ️ Selecciona variables diferentes para X e Y para generar el gráfico de densidad.")
    
    # Sección final con insights y conclusiones
    st.markdown("""
    <div style="background: linear-gradient(to right, #4364f7, #6fb1fc); padding: 15px; border-radius: 10px; margin: 20px 0;">
        <h3 style="color: white; margin: 0;">💡 Principales Hallazgos</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Insights automáticos basados en los datos filtrados
    st.markdown(f"""
    - El **precio promedio** de los diamantes seleccionados es de **${df_filtered['price'].mean():,.2f}**
    - La **correlación** entre quilates y precio es de **{df_filtered[['carat', 'price']].corr().iloc[0,1]:.2f}**
    - Los diamantes de corte '{df_filtered.groupby('cut')['price'].mean().idxmax()}' tienen el precio promedio más alto
    - Los diamantes de color '{df_filtered.groupby('color')['price'].mean().idxmax()}' son los más valiosos en promedio
    - La claridad tiene un impacto de **{abs(df_filtered.groupby('clarity')['price'].mean().max() - df_filtered.groupby('clarity')['price'].mean().min()) / df_filtered['price'].mean() * 100:.1f}%** en la variación de precios
    
    Continúa tu exploración en las siguientes secciones para ver los modelos predictivos.
    """)
    
    # Pie de página
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 0.9em;">
        Análisis exploratorio completado • Datos actualizados • Continúa con los modelos
    </p>
    """, unsafe_allow_html=True)

# =================== REGRESIÓN ===================
if seccion == "📈 Regresión":

    st.title("📈 Predicción Avanzada del Precio de Diamantes")

    # =================== Cargar Datos ===================
    df = sns.load_dataset("diamonds")

    # Convertir columnas categóricas a string
    categorical_cols = ['cut', 'color', 'clarity']
    df[categorical_cols] = df[categorical_cols].astype(str)

    # =================== Verificar si el Modelo Existe ===================
    modelo_path = "/tmp/model_regression.joblib"  # Ruta temporal en Streamlit Cloud
    modelo_drive_url = "https://drive.google.com/uc?id=1_BXt5mN391zac33WmvliAOKD7KalBzRe"

    if "modelo_regresion" not in st.session_state:
        if not os.path.exists(modelo_path):
            st.warning("⚠️ No se encontró un modelo guardado. Descargando desde Google Drive...")

            try:
                with st.spinner("Descargando modelo de regresión desde Google Drive..."):
                    gdown.download(modelo_drive_url, modelo_path, quiet=False)
                st.success("✅ Modelo descargado correctamente")
            except Exception as e:
                st.error(f"❌ Error al descargar el modelo: {e}")
                st.warning("⚠️ Se entrenará un nuevo modelo de regresión.")

                # =================== Preprocesamiento ===================
                X = df.drop(columns=["price"])
                y = df["price"]

                numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

                numerical_pipeline = make_pipeline(
                    SimpleImputer(strategy='median'),
                    StandardScaler()
                )

                categorical_pipeline = make_pipeline(
                    SimpleImputer(strategy='most_frequent'),
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                )

                preprocessor = make_column_transformer(
                    (numerical_pipeline, numerical_cols),
                    (categorical_pipeline, categorical_cols)
                )

                # =================== División de Datos ===================
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # =================== Entrenar Modelos ===================
                modelos = {
                    "Regresión Lineal": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
                }

                mejores_resultados = {}

                for nombre, modelo in modelos.items():
                    pipeline = make_pipeline(preprocessor, modelo)
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)

                    mejores_resultados[nombre] = {
                        "Modelo": pipeline,
                        "MAE": mae
                    }

                # =================== Seleccionar el Mejor Modelo ===================
                mejor_modelo_nombre = min(mejores_resultados, key=lambda x: mejores_resultados[x]["MAE"])
                mejor_modelo = mejores_resultados[mejor_modelo_nombre]["Modelo"]

                st.session_state.modelo_regresion = mejor_modelo

                # Guardar el modelo entrenado
                joblib.dump(mejor_modelo, modelo_path)
                st.success(f"✅ Modelo entrenado y guardado: {mejor_modelo_nombre}")

        # Cargar el modelo descargado
        try:
            with st.spinner("Cargando el modelo de predicción..."):
                st.session_state.modelo_regresion = joblib.load(modelo_path)
            st.success("✅ Modelo cargado correctamente")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {e}")
            st.stop()

    # =================== Página de Regresión ===================

    # Banner estilizado con animación y sombras
    st.markdown("""
    <div style="background: linear-gradient(to right, #3A86FF, #6fb1fc); padding: 30px; border-radius: 15px; margin-bottom: 20px; text-align: center; box-shadow: 0px 5px 15px rgba(0,0,0,0.2); animation: fadeIn 1.2s ease-in-out;">
        <h1 style="color: white; margin: 0; font-size: 36px; font-weight: bold;">💎 Estimación Inteligente del Precio de Diamantes</h1>
        <p style="color: #ffffffb3; font-size: 18px; margin-top: 10px;">Un análisis basado en Machine Learning para predecir con precisión el precio de un diamante</p>
    </div>
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    🔍 **Nuestro modelo de IA analiza múltiples factores del diamante y te ofrece la mejor estimación de precio.**

    💡 *Introduce los parámetros de tu diamante y descubre su valor de mercado.*
    """)

    # Sección de predicción con diseño premium
    st.markdown("### 🎯 Simulador de Predicción de Precios")

    with st.form("prediction_form"):
        st.markdown("<h4 style='color:#3A86FF;'>📌 Configuración del Diamante</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            carat = st.slider("💎 Quilates", min_value=0.2, max_value=5.0, value=1.0, step=0.1)
            depth = st.slider("📏 Profundidad (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
            table = st.slider("📐 Tabla (%)", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
            x = st.slider("📏 Longitud (x)", min_value=0.0, max_value=10.0, value=5.5, step=0.1)
            y = st.slider("📏 Ancho (y)", min_value=0.0, max_value=10.0, value=5.5, step=0.1)
            z = st.slider("📏 Altura (z)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        
        with col2:
            cut = st.selectbox("✨ Corte", df['cut'].unique())
            color = st.selectbox("🎨 Color", df['color'].unique())
            clarity = st.selectbox("🔍 Claridad", df['clarity'].unique())
        
        st.markdown("---")
        predict_button = st.form_submit_button("🚀 Predecir Precio")
        
    if predict_button:
        try:
            # Recuperar el modelo entrenado desde session_state
            modelo = st.session_state.modelo_regresion
            
            input_data = pd.DataFrame({
                'carat': [carat],
                'depth': [depth],
                'table': [table],
                'x': [x],
                'y': [y],
                'z': [z],
                'cut': [cut],
                'color': [color],
                'clarity': [clarity]
            })

            with st.spinner("🧐 Calculando precio..."):
                time.sleep(1)
                precio_predicho = modelo.predict(input_data)[0]

            # =================== Comparaciones con Datos Reales ===================
            avg_price = df["price"].mean()
            avg_price_per_carat = df.groupby("carat")["price"].mean()
            avg_price_per_cut = df.groupby("cut")["price"].mean()

            if cut in avg_price_per_cut.index:
                price_for_closest_cut = avg_price_per_cut.loc[cut]
            else:
                price_for_closest_cut = avg_price

            closest_carat = avg_price_per_carat.index[np.abs(avg_price_per_carat.index - carat).argmin()]
            price_for_closest_carat = avg_price_per_carat.loc[closest_carat]

            # =================== Mostrar Resultados ===================
            st.markdown("<h3 style='color:#3A86FF;'>📊 Resultados de la Predicción</h3>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: #eaf4ff; padding: 30px; border-radius: 15px; border-left: 8px solid #3A86FF; text-align: center; box-shadow: 0px 4px 12px rgba(0,0,0,0.1);">
                <h2 style="color: #3A86FF; margin-top: 0; font-size: 36px;">💰 Precio Estimado: ${precio_predicho:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📈 Precio Promedio", f"${avg_price:,.2f}")

            with col2:
                st.metric(f"📊 Precio Medio para {closest_carat} Quilates", f"${price_for_closest_carat:,.2f}")

            with col3:
                st.metric(f"🔍 Precio Medio para {cut} Cut", f"${price_for_closest_cut:,.2f}")

        except Exception as e:
            st.error(f"❌ Error al realizar la predicción: {e}")

# =================== CLASIFICACIÓN ===================
if seccion == "⚡ Clasificación":
    
    st.title("⚡ Predicción del Corte del Diamante")

    # =================== Cargar Datos ===================
    df = sns.load_dataset("diamonds")

    # Convertir columnas categóricas a string
    categorical_cols = ['color', 'clarity']
    df[categorical_cols] = df[categorical_cols].astype(str)

    # =================== Verificar si el Modelo Existe ===================
    modelo_path = "model_classification.joblib"
    drive_url = "https://drive.google.com/uc?id=1O7E7Q4u3bn4AuVn5tkIizLhgtDnqTBew"  # ID extraído del enlace compartido

    if "modelo_clasificacion" not in st.session_state:
        if os.path.exists(modelo_path):
            try:
                with st.spinner("Cargando el modelo de clasificación..."):
                    st.session_state.modelo_clasificacion = joblib.load(modelo_path)
                st.success("✅ Modelo cargado correctamente")
            except Exception as e:
                st.error(f"❌ Error al cargar el modelo: {e}")
        else:
            st.warning("⚠️ No se encontró un modelo guardado. Intentando descargar desde Google Drive...")

            try:
                gdown.download(drive_url, modelo_path, quiet=False)
                with st.spinner("Cargando el modelo de clasificación..."):
                    st.session_state.modelo_clasificacion = joblib.load(modelo_path)
                st.success("✅ Modelo descargado y cargado correctamente desde Google Drive")
            except Exception as e:
                st.error(f"❌ Error al descargar el modelo: {e}. Se procederá a entrenar uno nuevo.")

                # =================== Preprocesamiento ===================
                X = df.drop(columns=["cut"])  # 'cut' es lo que queremos predecir
                y = df["cut"]

                numerical_cols = ["carat", "depth", "table", "x", "y", "z", "price"]

                categorical_pipeline = make_pipeline(
                    OneHotEncoder(handle_unknown='ignore')
                )
                numerical_pipeline = make_pipeline(
                    StandardScaler()
                )

                column_transformer = make_column_transformer(
                    (numerical_pipeline, numerical_cols),
                    (categorical_pipeline, ["color", "clarity"])
                )

                # =================== División de Datos ===================
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                # =================== Entrenar Modelo ===================
                modelo = make_pipeline(column_transformer, RandomForestClassifier(n_estimators=100, random_state=42))
                modelo.fit(X_train, y_train)

                # Evaluación
                y_pred = modelo.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.session_state.modelo_clasificacion = modelo
                joblib.dump(modelo, modelo_path)

                st.success(f"✅ Modelo entrenado y guardado con precisión: {accuracy:.4f}")

    # =================== Página de Clasificación ===================

    st.markdown("""
    <div style="background: linear-gradient(to right, #FF6B6B, #FF8E8E); padding: 30px; border-radius: 15px; 
    margin-bottom: 20px; text-align: center; box-shadow: 0px 5px 15px rgba(0,0,0,0.2); animation: fadeIn 1.2s ease-in-out;">
        <h1 style="color: white; margin: 0; font-size: 36px; font-weight: bold;">🔍 Clasificación de Calidad del Corte</h1>
        <p style="color: #ffffffb3; font-size: 18px; margin-top: 10px;">Un modelo de Machine Learning para predecir el tipo de corte de un diamante</p>
    </div>
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    🔍 **Este modelo de clasificación predice la calidad del corte de un diamante en función de sus características.**

    💡 *Introduce los atributos del diamante y obtén su clasificación.*
    """)

    # Sección de predicción con diseño premium
    st.markdown("### 🎯 Simulador de Clasificación del Corte")

    with st.form("classification_form"):
        st.markdown("<h4 style='color:#FF6B6B;'>📌 Características del Diamante</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            carat = st.slider("💎 Quilates", min_value=0.2, max_value=5.0, value=1.0, step=0.1)
            depth = st.slider("📏 Profundidad (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
            table = st.slider("📐 Tabla (%)", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
            x = st.slider("📏 Longitud (x)", min_value=0.0, max_value=10.0, value=5.5, step=0.1)
            y = st.slider("📏 Ancho (y)", min_value=0.0, max_value=10.0, value=5.5, step=0.1)
            z = st.slider("📏 Altura (z)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
            price = st.number_input("💰 Precio del Diamante ($)", min_value=100, max_value=20000, value=5000, step=50)
        
        with col2:
            color = st.selectbox("🎨 Color", df['color'].unique())
            clarity = st.selectbox("🔍 Claridad", df['clarity'].unique())
        
        st.markdown("---")
        classify_button = st.form_submit_button("⚡ Clasificar Corte")
        
    if classify_button:
        try:
            # Recuperar el modelo entrenado
            modelo_clasificacion = st.session_state.modelo_clasificacion
            # Crear DataFrame con los datos ingresados por el usuario
            input_data = pd.DataFrame({
                'carat': [carat],
                'depth': [depth],
                'table': [table],
                'x': [x],
                'y': [y],
                'z': [z],
                'price': [price],  # 🚀 Ahora agregamos el precio
                'color': [color],
                'clarity': [clarity]
            })

            # Asegurar que las columnas coinciden con las usadas en el entrenamiento
            expected_columns = modelo_clasificacion.feature_names_in_
            missing_cols = set(expected_columns) - set(input_data.columns)

            if missing_cols:
                st.error(f"❌ Error al realizar la clasificación: columnas faltantes: {missing_cols}")
            else:
                # Realizar la predicción
                with st.spinner("🧐 Clasificando el corte del diamante..."):
                    time.sleep(1)
                    corte_predicho = modelo_clasificacion.predict(input_data)[0]
                    proba_predicho = modelo_clasificacion.predict_proba(input_data)

                # =================== Mostrar Resultados ===================
                st.markdown("<h3 style='color:#FF6B6B;'>📊 Resultado de la Clasificación</h3>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background-color: #ffe6e6; padding: 30px; border-radius: 15px; border-left: 8px solid #FF6B6B; 
                text-align: center; box-shadow: 0px 4px 12px rgba(0,0,0,0.1);">
                    <h2 style="color: #FF6B6B; margin-top: 0; font-size: 36px;">🔍 Corte Predicho: {corte_predicho}</h2>
                </div>
                """, unsafe_allow_html=True)

                # Mostrar probabilidades de predicción
                st.markdown("### 🔬 Probabilidades de Clasificación")
                df_proba = pd.DataFrame(proba_predicho, columns=modelo_clasificacion.classes_)
                st.bar_chart(df_proba.T)

                # Comparación con la base de datos
                st.markdown("### 📊 Distribución de Cortes en la Base de Datos")
                corte_counts = df['cut'].value_counts(normalize=True) * 100

                fig = px.bar(
                    x=corte_counts.index,
                    y=corte_counts.values,
                    text=[f"{val:.2f}%" for val in corte_counts.values],
                    labels={'x': 'Corte', 'y': 'Frecuencia (%)'},
                    title="Distribución de Cortes en la Base de Datos",
                    color=corte_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set1
                )

                fig.update_traces(textposition='outside')
                fig.update_layout(height=400)

                st.plotly_chart(fig, use_container_width=True)

                # Mostrar interpretación de las variables
                st.markdown("### 🧠 Factores Claves en la Predicción")
                st.markdown(f"""
                - **Quilates (Carat)**: El tamaño del diamante influye en su clasificación.
                - **Precio**: Diamantes más caros suelen tener cortes de mayor calidad.
                - **Color y Claridad**: Factores estéticos que pueden impactar la clasificación.
                - **Proporciones (Profundidad, Tabla, Dimensiones)**: Determinan cómo refleja la luz el diamante.
                """)

        except Exception as e:
            st.error(f"❌ Error al realizar la clasificación: {e}")
            
# =================== SIMULADOR ===================
if seccion == "🧪 Simulador":
    st.title("🧪 Simulador Inteligente de Diamantes")

    st.markdown("""
    <div style="background: linear-gradient(to right, #6A11CB, #2575FC); padding: 30px; border-radius: 15px; 
    margin-bottom: 20px; text-align: center; box-shadow: 0px 5px 15px rgba(0,0,0,0.2); animation: fadeIn 1.2s ease-in-out;">
        <h1 style="color: white; margin: 0; font-size: 36px; font-weight: bold;">🔮 Predicción de Precio y Corte</h1>
        <p style="color: #ffffffb3; font-size: 18px; margin-top: 10px;">
            Un simulador avanzado que predice el precio y la calidad del corte de un diamante en base a sus características.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    🔍 **Introduce las características del diamante y descubre su precio y calidad estimada.**  
    """)

    # =================== Verificar si los modelos existen ===================
    modelo_reg_path = "/tmp/model_regression.joblib"
    modelo_clas_path = "/tmp/model_classification.joblib"

    modelo_reg_url = "https://drive.google.com/uc?id=1_BXt5mN391zac33WmvliAOKD7KalBzRe"
    modelo_clas_url = "https://drive.google.com/uc?id=1O7E7Q4u3bn4AuVn5tkIizLhgtDnqTBew"

    if "modelo_regresion" not in st.session_state:
        if not os.path.exists(modelo_reg_path):
            st.warning("⚠️ No se encontró un modelo de regresión guardado. Descargando desde Google Drive...")
            try:
                with st.spinner("Descargando modelo de regresión..."):
                    gdown.download(modelo_reg_url, modelo_reg_path, quiet=False)
                st.success("✅ Modelo de regresión descargado correctamente")
            except Exception as e:
                st.error(f"❌ Error al descargar el modelo de regresión: {e}")

        try:
            with st.spinner("Cargando el modelo de regresión..."):
                st.session_state.modelo_regresion = joblib.load(modelo_reg_path)
            st.success("✅ Modelo de regresión cargado correctamente")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo de regresión: {e}")
            st.stop()

    if "modelo_clasificacion" not in st.session_state:
        if not os.path.exists(modelo_clas_path):
            st.warning("⚠️ No se encontró un modelo de clasificación guardado. Descargando desde Google Drive...")
            try:
                with st.spinner("Descargando modelo de clasificación..."):
                    gdown.download(modelo_clas_url, modelo_clas_path, quiet=False)
                st.success("✅ Modelo de clasificación descargado correctamente")
            except Exception as e:
                st.error(f"❌ Error al descargar el modelo de clasificación: {e}")

        try:
            with st.spinner("Cargando el modelo de clasificación..."):
                st.session_state.modelo_clasificacion = joblib.load(modelo_clas_path)
            st.success("✅ Modelo de clasificación cargado correctamente")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo de clasificación: {e}")
            st.stop()

    # Formulario interactivo
    with st.form("simulator_form"):
        st.markdown("<h4 style='color:#6A11CB;'>📌 Configuración del Diamante</h4>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            carat = st.slider("💎 Quilates", min_value=0.2, max_value=5.0, value=1.0, step=0.1)
            depth = st.slider("📏 Profundidad (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
            table = st.slider("📐 Tabla (%)", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
            x = st.slider("📏 Longitud (x)", min_value=0.0, max_value=10.0, value=5.5, step=0.1)
            y = st.slider("📏 Ancho (y)", min_value=0.0, max_value=10.0, value=5.5, step=0.1)
            z = st.slider("📏 Altura (z)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
            price = st.number_input("💰 Precio del Diamante ($)", min_value=100, max_value=20000, value=5000, step=50)

        with col2:
            cut = st.selectbox("✨ Corte", df['cut'].unique())
            color = st.selectbox("🎨 Color", df['color'].unique())
            clarity = st.selectbox("🔍 Claridad", df['clarity'].unique())

        st.markdown("---")
        simulate_button = st.form_submit_button("🔮 Ejecutar Simulación")

    if simulate_button:
        try:
            modelo_regresion = st.session_state.modelo_regresion
            modelo_clasificacion = st.session_state.modelo_clasificacion
            input_data = pd.DataFrame({
                'carat': [carat],
                'depth': [depth],
                'table': [table],
                'x': [x],
                'y': [y],
                'z': [z],
                'price': [price],  # 🚀 Agregado para clasificación
                'cut': [cut],  # Agregado para predicción de precio
                'color': [color],
                'clarity': [clarity]
            })

            # =================== Predicción del Precio ===================
            with st.spinner("Calculando precio..."):
                time.sleep(1)
                precio_predicho = modelo_regresion.predict(input_data)[0]

            # =================== Predicción del Corte ===================
            with st.spinner("Clasificando el corte..."):
                time.sleep(1)
                corte_predicho = modelo_clasificacion.predict(input_data)[0]
                proba_predicho = modelo_clasificacion.predict_proba(input_data)

            # =================== Mostrar Resultados ===================
            st.markdown("<h3 style='color:#6A11CB;'>📊 Resultados de la Simulación</h3>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div style="background-color: #E3E3FF; padding: 30px; border-radius: 15px; border-left: 8px solid #6A11CB; 
                text-align: center; box-shadow: 0px 4px 12px rgba(0,0,0,0.1);">
                    <h2 style="color: #6A11CB; margin-top: 0; font-size: 36px;">💰 Precio Estimado: ${:,.2f}</h2>
                </div>
                """.format(precio_predicho), unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style="background-color: #FFE6E6; padding: 30px; border-radius: 15px; border-left: 8px solid #FF6B6B; 
                text-align: center; box-shadow: 0px 4px 12px rgba(0,0,0,0.1);">
                    <h2 style="color: #FF6B6B; margin-top: 0; font-size: 36px;">🔍 Corte Predicho: {}</h2>
                </div>
                """.format(corte_predicho), unsafe_allow_html=True)

            # Mostrar probabilidades de clasificación
            st.markdown("### 🔬 Probabilidades de Clasificación")
            df_proba = pd.DataFrame(proba_predicho, columns=modelo_clasificacion.classes_)
            st.bar_chart(df_proba.T)

            # Explicación adicional
            st.markdown("### 🧠 Factores Claves en la Predicción")
            st.markdown(f"""
            - **Quilates (Carat)**: Afecta tanto el precio como la clasificación del corte.
            - **Precio**: Un mayor precio generalmente indica cortes de mayor calidad.
            - **Color y Claridad**: Influye en la percepción del corte.
            - **Proporciones (Profundidad, Tabla, Dimensiones)**: Determinan cómo refleja la luz el diamante.
            """)

        except Exception as e:
            st.error(f"❌ Error en la simulación: {e}")