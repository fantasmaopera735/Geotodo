# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- CONFIGURACIÓN DE LA RUTA_RELATIVA ---
# Asegúrate de que este archivo esté en la misma carpeta que Geotodo.csv
RUTA_CSV = 'Geotodo.csv' 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Geotodo - Buscador de Patrones",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Geotodo - Buscador de Patrones Consecutivos")
st.markdown("""
Esta herramienta busca un número específico y analiza qué salió en los **sorteos posteriores**.
Ideal para encontrar números que "arrastran" a otros.
""")

# --- FUNCIÓN PARA CARGAR DATOS ---
@st.cache_resource
def cargar_datos_geotodo(_ruta_csv):
    try:
        if not os.path.exists(_ruta_csv):
            st.error(f"❌ Error: No se encontró el archivo {RUTA_CSV}.")
            st.stop()
        
        # Lectura básica del CSV
        df = pd.read_csv(_ruta_csv, sep=';', encoding='latin-1')
        
        # Normalización de columnas
        df.rename(columns={
            'Fecha': 'Fecha',
            'Tarde/Noche': 'Tipo_Sorteo', 
            'Fijo': 'Fijo',
            '1er Corrido': 'Primer_Corrido',
            '2do Corrido': 'Segundo_Corrido'
        }, inplace=True)
        
        # Convertir fechas
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)
        
        # Normalizar Tipo de Sorteo
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({
            'MAÑANA': 'M', 'M': 'M', 'MANANA': 'M',
            'TARDE': 'T', 'T': 'T',
            'NOCHE': 'N', 'N': 'N'
        }).fillna('OTRO')

        # --- IMPORTANTE: Solo analizamos el FIJO para esta búsqueda de patrones ---
        # Filtramos para tener una lista limpia y cronológica de FIJOS
        df_fijos = df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy()
        df_fijos = df_fijos.rename(columns={'Fijo': 'Numero'})
        df_fijos['Numero'] = pd.to_numeric(df_fijos['Numero'], errors='coerce').astype(int)
        
        # Orden cronológico estricto (Mañana -> Tarde -> Noche)
        draw_order_map = {'M': 0, 'T': 1, 'N': 2}
        df_fijos['draw_order'] = df_fijos['Tipo_Sorteo'].map(draw_order_map).fillna(3)
        df_fijos['sort_key'] = df_fijos['Fecha'] + pd.to_timedelta(df_fijos['draw_order'], unit='h')
        df_fijos = df_fijos.sort_values(by='sort_key').reset_index(drop=True)
        
        return df_fijos
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

# --- FUNCIÓN PRINCIPAL DE BÚSQUEDA ---
def analizar_siguientes(df_fijos, numero_busqueda, ventana_sorteos):
    """
    Busca el numero_busqueda en el historial y cuenta qué números
    aparecen en los 'ventana_sorteos' siguientes.
    """
    # 1. Encontrar todos los índices donde aparece el número disparador
    indices_disparador = df_fijos[df_fijos['Numero'] == numero_busqueda].index.tolist()
    
    if not indices_disparador:
        return None, 0 # No se encontró el número
    
    lista_siguientes = []
    
    total_analizados = 0
    
    # 2. Para cada aparición, mirar la ventana posterior
    for idx in indices_disparador:
        # Calculamos el rango de la ventana (evitando pasarnos del final del DF)
        inicio = idx + 1
        fin = idx + ventana_sorteos + 1
        
        if inicio >= len(df_fijos):
            continue
            
        # Extraer los números de la ventana
        ventana = df_fijos.iloc[inicio:fin]['Numero'].tolist()
        lista_siguientes.extend(ventana)
        total_analizados += 1
        
    # 3. Contar frecuencias
    conteo = Counter(lista_siguientes)
    
    # 4. Convertir a DataFrame para mostrar
    df_resultado = pd.DataFrame.from_dict(conteo, orient='index', columns=['Frecuencia'])
    df_resultado.index.name = 'Número'
    df_resultado.reset_index(inplace=True)
    
    # --- FILTRO CLAVE: Solo mostrar los que sí salieron (Frecuencia > 0) ---
    df_resultado = df_resultado[df_resultado['Frecuencia'] > 0]
    
    # Ordenar de mayor a menor
    df_resultado = df_resultado.sort_values(by='Frecuencia', ascending=False).reset_index(drop=True)
    
    # Calcular porcentaje respecto al total de "espacios" analizados
    total_slots = len(lista_siguientes)
    if total_slots > 0:
        df_resultado['Probabilidad (%)'] = (df_resultado['Frecuencia'] / total_slots * 100).round(2)
    else:
        df_resultado['Probabilidad (%)'] = 0
    
    # Formatear número a 2 dígitos
    df_resultado['Número'] = df_resultado['Número'].apply(lambda x: f"{x:02d}")
    
    return df_resultado, total_analizados

# --- EJECUCIÓN DE LA APP ---
def main():
    # Cargar datos
    df_fijos = cargar_datos_geotodo(RUTA_CSV)
    
    st.sidebar.header("⚙️ Configuración de Búsqueda")
    
    # Input del número disparador
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        numero_busqueda = st.number_input(
            "Número Disparador (0-99):", 
            min_value=0, max_value=99, value=40, step=1, 
            format="%02d"
        )
    
    # Input de la ventana de búsqueda
    with col_b:
        ventana_sorteos = st.slider(
            "Buscar en los próximos X sorteos:", 
            min_value=1, max_value=30, value=15, step=1
        )
    
    st.sidebar.info(f"Buscaré qué salió después del **{numero_busqueda:02d}** en sus siguientes **{ventana_sorteos}** sorteos.")
    
    # Botón de acción
    buscar_btn = st.button("🔎 Iniciar Análisis de Patrones", type="primary", use_container_width=True)
    
    if buscar_btn or 'ultima_busqueda' in st.session_state:
        # Mantener estado para que no desaparezca al interactuar con gráficos
        if buscar_btn:
            st.session_state['ultima_busqueda'] = {'num': numero_busqueda, 'ventana': ventana_sorteos}
        
        params = st.session_state.get('ultima_busqueda', {'num': numero_busqueda, 'ventana': ventana_sorteos})
        num = params['num']
        ven = params['ventana']
        
        st.markdown("---")
        
        st.subheader(f"Análisis de Patrones para el número: 🎯 **{num:02d}**")
        
        # Ejecutar lógica
        df_resultado, total_ocurrencias = analizar_siguientes(df_fijos, num, ven)
        
        if df_resultado is None:
            st.error(f"❌ El número **{num:02d}** nunca ha salido en el historial cargado.")
        else:
            st.success(f"✅ Se encontraron **{total_ocurrencias}** oportunidades históricas del número {num:02d}.")
            st.info(f"📊 Se analizaron en total **{len(df_resultado)}** números distintos que aparecieron después de él.")
            
            # --- SOLUCIÓN AL ERROR INT64 ---
            # Convertimos el valor máximo a un entero nativo de Python antes de pasarlo a la configuración
            max_freq_val = int(df_resultado['Frecuencia'].max())

            st.dataframe(
                df_resultado.head(30), # Mostrar top 30 para no saturar
                column_config={
                    "Número": st.column_config.TextColumn("Número", width="small"),
                    "Frecuencia": st.column_config.ProgressColumn(
                        "Frecuencia", 
                        help="Cuántas veces salió", 
                        format="%d", 
                        min_value=0, 
                        max_value=max_freq_val # <--- Aquí está la corrección
                    ),
                    "Probabilidad (%)": st.column_config.NumberColumn("Probabilidad (%)", format="%.2f%%")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Gráfico visual de los Top 10
            st.markdown("---")
            st.subheader("📈 Top 10 Números Más Frecuentes")
            
            df_top10 = df_resultado.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x='Frecuencia', 
                y='Número', 
                data=df_top10, 
                palette='viridis', 
                ax=ax
            )
            ax.set_title(f"Números que más siguen al {num:02d} (próximos {ven} sorteos)")
            ax.set_xlabel("Frecuencia de Aparición")
            st.pyplot(fig)
            
            # Análisis textual simple
            if len(df_resultado) > 0:
                numero_mas_probable = df_resultado.iloc[0]['Número']
                probabilidad_alta = df_resultado.iloc[0]['Probabilidad (%)']
                
                st.markdown(f"""
                <div style="background-color:#E8F5E9; padding:15px; border-radius:10px; border-left: 5px solid #4CAF50;">
                    <h4>💡 Insight Rápido:</h4>
                    Históricamente, cuando sale el <b>{num:02d}</b>, el número <b>{numero_mas_probable}</b> 
                    aparece en los próximos {ven} sorteos con una probabilidad del <b>{probabilidad_alta}%</b>.
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()