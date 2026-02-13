# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from collections import Counter
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE LA RUTA_RELATIVA ---
RUTA_CSV = 'flotodo.csv' 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Flotodo - Suite Completa",
    page_icon="🚤",
    layout="wide"
)

st.title("🚤 Flotodo - Suite de Análisis Avanzado")

# --- FUNCIÓN PARA CARGAR DATOS ---
@st.cache_resource
def cargar_datos_flotodo(_ruta_csv):
    try:
        if not os.path.exists(_ruta_csv):
            st.error(f"❌ Error: No se encontró el archivo {_ruta_csv}.")
            st.stop()
        
        df = pd.read_csv(_ruta_csv, sep=';', encoding='latin-1')
        
        df.rename(columns={
            'Fecha': 'Fecha',
            'Tarde/Noche': 'Tipo_Sorteo', 
            'Fijo': 'Fijo',
            '1er Corrido': 'Primer_Corrido',
            '2do Corrido': 'Segundo_Corrido'
        }, inplace=True)
        
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)
        
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({
            'MAÑANA': 'M', 'M': 'M', 'MANANA': 'M',
            'TARDE': 'T', 'T': 'T',
            'NOCHE': 'N', 'N': 'N'
        }).fillna('OTRO')

        df_fijos = df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy()
        df_fijos = df_fijos.rename(columns={'Fijo': 'Numero'})
        df_fijos['Numero'] = pd.to_numeric(df_fijos['Numero'], errors='coerce').astype(int)
        
        # Orden cronológico estricto
        draw_order_map = {'M': 0, 'T': 1, 'N': 2}
        df_fijos['draw_order'] = df_fijos['Tipo_Sorteo'].map(draw_order_map).fillna(3)
        df_fijos['sort_key'] = df_fijos['Fecha'] + pd.to_timedelta(df_fijos['draw_order'], unit='h')
        df_fijos = df_fijos.sort_values(by='sort_key').reset_index(drop=True)
        
        return df_fijos
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

# --- FUNCIÓN 1: PATRONES (NÚMEROS) ---
def analizar_siguientes(df_fijos, numero_busqueda, ventana_sorteos):
    indices_disparador = df_fijos[df_fijos['Numero'] == numero_busqueda].index.tolist()
    if not indices_disparador: return None, 0 
    
    lista_siguientes = []
    total_analizado = 0
    for idx in indices_disparador:
        inicio = idx + 1
        fin = idx + ventana_sorteos + 1
        if inicio >= len(df_fijos): continue
        ventana = df_fijos.iloc[inicio:fin]['Numero'].tolist()
        lista_siguientes.extend(ventana)
        total_analizado += 1
        
    conteo = Counter(lista_siguientes)
    df_res = pd.DataFrame.from_dict(conteo, orient='index', columns=['Frecuencia'])
    df_res.index.name = 'Número'
    df_res.reset_index(inplace=True)
    df_res = df_res[df_res['Frecuencia'] > 0]
    df_res = df_res.sort_values(by='Frecuencia', ascending=False).reset_index(drop=True)
    
    total_slots = len(lista_siguientes)
    df_res['Probabilidad (%)'] = (df_res['Frecuencia'] / total_slots * 100).round(2) if total_slots > 0 else 0
    df_res['Número'] = df_res['Número'].apply(lambda x: f"{x:02d}")
    return df_res, total_analizado

# --- FUNCIÓN 2: ALMANAQUE ---
def analizar_almanaque(df_fijos, quincena, meses_atras):
    fecha_fin = datetime.now()
    fecha_inicio = fecha_fin - pd.DateOffset(months=meses_atras)
    df_filtrado = df_fijos[(df_fijos['Fecha'] >= fecha_inicio) & (df_fijos['Fecha'] <= fecha_fin)].copy()
    if df_filtrado.empty: return None, None, None
    
    df_filtrado['Dia'] = df_filtrado['Fecha'].dt.day
    if quincena == "1ra Quincena (Días 1 al 15)":
        df_filtrado = df_filtrado[df_filtrado['Dia'] <= 15]
    else:
        df_filtrado = df_filtrado[df_filtrado['Dia'] > 15]
    
    if df_filtrado.empty: return None, None, None

    df_filtrado['Decena'] = df_filtrado['Numero'] // 10
    df_filtrado['Unidad'] = df_filtrado['Numero'] % 10
    
    cnt_dec = df_filtrado['Decena'].value_counts().reindex(range(10), fill_value=0)
    cnt_uni = df_filtrado['Unidad'].value_counts().reindex(range(10), fill_value=0)
    
    return df_filtrado, cnt_dec, cnt_uni

# --- FUNCIÓN 3: GENERADOR DE PROPUESTA INTELIGENTE ---
def generar_sugerencia_tendencia(df_fijos, dias_tendencia, dias_minimo_gap):
    fecha_hoy = datetime.now()
    fecha_inicio_tendencia = fecha_hoy - timedelta(days=dias_tendencia)
    
    df_tendencia = df_fijos[df_fijos['Fecha'] >= fecha_inicio_tendencia].copy()
    
    if df_tendencia.empty:
        return pd.DataFrame()
    
    df_tendencia['Decena'] = df_tendencia['Numero'] // 10
    df_tendencia['Unidad'] = df_tendencia['Numero'] % 10
    
    top_decenas = df_tendencia['Decena'].value_counts().head(3).index.tolist()
    top_unidades = df_tendencia['Unidad'].value_counts().head(3).index.tolist()
    
    st.info(f"🔥 Tendencia Detectada (Últimos {dias_tendencia} días): Decenas {top_decenas} | Unidades {top_unidades}")
    
    candidatos = []
    for d in top_decenas:
        for u in top_unidades:
            candidatos.append(d * 10 + u)
            
    resultados = []
    
    for num in candidatos:
        df_num = df_fijos[df_fijos['Numero'] == num]
        
        if df_num.empty:
            gap = (fecha_hoy - df_fijos['Fecha'].min()).days
            ultima_fecha = "Nunca"
        else:
            ultima_fecha = df_num['Fecha'].max()
            gap = (fecha_hoy - ultima_fecha).days
        
        if gap >= dias_minimo_gap:
            decena = num // 10
            unidad = num % 10
            estado = "⚡ Muy Oportuno" if gap > (dias_minimo_gap * 1.5) else "✅ Oportuno"
            
            resultados.append({
                'Número': f"{num:02d}",
                'Gap (Días)': gap,
                'Última Salida': ultima_fecha.strftime('%d/%m/%Y'),
                'Combinación': f"Dec {decena} + Uni {unidad}",
                'Estado': estado
            })
            
    return pd.DataFrame(resultados).sort_values(by='Gap (Días)', ascending=False).reset_index(drop=True)

# --- FUNCIÓN 4: BÚSQUEDA DE SECUENCIA DE DÍGITOS (CON FECHAS) ---
def buscar_secuencia_digitos(df_fijos, secuencia_input, tipo_digito):
    """
    Busca secuencia y devuelve el dígito siguiente + ejemplos de números CON FECHA.
    """
    try:
        patron = [int(x.strip()) for x in secuencia_input.replace(',', ' ').split() if x.strip().isdigit()]
    except:
        return None, "Error en el formato. Usa números del 0 al 9 separados por espacios o comas."
    
    if len(patron) == 0:
        return None, "La secuencia está vacía."
    if len(patron) > 5:
        return None, "La secuencia máxima permitida es de 5 dígitos."
    if any(x < 0 or x > 9 for x in patron):
        return None, "Los dígitos deben estar entre 0 y 9."

    # Extraer lista histórica de dígitos
    if tipo_digito == "Decena":
        lista_historica = (df_fijos['Numero'] // 10).tolist()
    else:
        lista_historica = (df_fijos['Numero'] % 10).tolist()
        
    long_patron = len(patron)
    
    # Diccionario para guardar datos: {digito_siguiente: {'count': 0, 'ejemplos': []}}
    datos_resultado = {}
    
    for i in range(len(lista_historica) - long_patron):
        sub_lista = lista_historica[i : i + long_patron]
        if sub_lista == patron:
            siguiente_digito = lista_historica[i + long_patron]
            
            # Extraer información de la fila donde ocurrió el evento (número + fecha)
            row_evento = df_fijos.iloc[i + long_patron]
            numero_completo = row_evento['Numero']
            fecha_evento = row_evento['Fecha']
            fecha_str = fecha_evento.strftime('%d/%m/%Y')
            
            if siguiente_digito not in datos_resultado:
                datos_resultado[siguiente_digito] = {'count': 0, 'ejemplos': []}
            
            datos_resultado[siguiente_digito]['count'] += 1
            
            # Guardar hasta 3 ejemplos únicos con formato "Número (Fecha)"
            # Ejemplo: "52 (15/10/2023)"
            if len(datos_resultado[siguiente_digito]['ejemplos']) < 3:
                formato_ejemplo = f"{numero_completo:02d} ({fecha_str})"
                if formato_ejemplo not in datos_resultado[siguiente_digito]['ejemplos']:
                    datos_resultado[siguiente_digito]['ejemplos'].append(formato_ejemplo)
            
    if not datos_resultado:
        return None, f"El patrón {patron} no se ha repetido consecutivamente en el historial."
    
    # Construir DataFrame
    rows = []
    for digito, data in datos_resultado.items():
        rows.append({
            'Dígito Siguiente': digito,
            'Frecuencia': data['count'],
            'Ejemplos Históricos (Núm + Fecha)': ", ".join(data['ejemplos']),
            'Probabilidad (%)': 0 # Se calcula abajo
        })
        
    df_res = pd.DataFrame(rows)
    total = df_res['Frecuencia'].sum()
    df_res['Probabilidad (%)'] = (df_res['Frecuencia'] / total * 100).round(2)
    df_res = df_res.sort_values(by='Frecuencia', ascending=False).reset_index(drop=True)
    
    return df_res, None

# --- EJECUCIÓN DE LA APP ---
def main():
    df_fijos = cargar_datos_flotodo(RUTA_CSV)
    
    st.sidebar.header("⚙️ Panel de Control")
    
    # --- INFO DE DATOS ---
    with st.sidebar.expander("📂 Información del Archivo", expanded=True):
        df_tarde_info = df_fijos[df_fijos['Tipo_Sorteo'] == 'T']
        df_noche_info = df_fijos[df_fijos['Tipo_Sorteo'] == 'N']
        
        st.markdown("**Últimos Sorteos Cargados:**")
        if not df_tarde_info.empty:
            ultima_tarde = df_tarde_info['Fecha'].max()
            num_tarde = df_tarde_info[df_tarde_info['Fecha'] == ultima_tarde]['Numero'].values[0]
            st.metric("🌞 Última Tarde", f"{ultima_tarde.strftime('%d/%m/%Y')}", delta=f"Fijo: {num_tarde:02d}")
        else:
            st.warning("Sin datos de Tarde")

        if not df_noche_info.empty:
            ultima_noche = df_noche_info['Fecha'].max()
            num_noche = df_noche_info[df_noche_info['Fecha'] == ultima_noche]['Numero'].values[0]
            st.metric("🌙 Última Noche", f"{ultima_noche.strftime('%d/%m/%Y')}", delta=f"Fijo: {num_noche:02d}")
        else:
            st.warning("Sin datos de Noche")

    # --- AGREGAR SORTEO ---
    with st.sidebar.expander("📝 Agregar Nuevo Sorteo", expanded=False):
        fecha_nueva = st.date_input("Fecha:", value=datetime.now().date(), format="DD/MM/YYYY", label_visibility="collapsed")
        sesion = st.radio("Sesión:", ["Tarde (T)", "Noche (N)"], horizontal=True, label_visibility="collapsed")
        
        col_a, col_b = st.columns(2)
        with col_a:
            fijo = st.number_input("Fijo", min_value=0, max_value=99, value=0, format="%02d", label_visibility="collapsed")
        with col_b:
            c1 = st.number_input("1er Corr.", min_value=0, max_value=99, value=0, format="%02d", label_visibility="collapsed")
        p2 = st.number_input("2do Corr.", min_value=0, max_value=99, value=0, format="%02d", label_visibility="collapsed")
        
        if st.button("💾 Guardar", type="primary", use_container_width=True):
            sesion_code = sesion.split('(')[1].replace(')', '') 
            fecha_str = fecha_nueva.strftime('%d/%m/%Y')
            linea_nueva = f"{fecha_str};{sesion_code};{fijo};{c1};{p2}\n"
            try:
                with open(RUTA_CSV, 'a', encoding='latin-1') as f:
                    f.write(linea_nueva)
                st.success("✅ Guardado. Recargando...")
                st.cache_resource.clear()
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # --- BOTÓN FORZAR RECARGA ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Forzar Recarga de Datos"):
        st.cache_resource.clear()
        st.sidebar.success("Caché limpio. Recargando...")
        time.sleep(1)
        st.rerun()

    # --- SELECCIÓN DE SESIÓN ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Filtro de Análisis")
    modo_sesion = st.sidebar.radio("Modo:", ["General", "Solo Tarde (T)", "Solo Noche (N)"])

    if "Tarde" in modo_sesion:
        df_analisis = df_fijos[df_fijos['Tipo_Sorteo'] == 'T'].copy()
        titulo_sesion = "Tarde"
    elif "Noche" in modo_sesion:
        df_analisis = df_fijos[df_fijos['Tipo_Sorteo'] == 'N'].copy()
        titulo_sesion = "Noche"
    else:
        df_analisis = df_fijos.copy()
        titulo_sesion = "General"

    if df_analisis.empty:
        st.warning(f"No hay registros para: **{modo_sesion}**")
        st.stop()

    tab_patrones, tab_almanaque, tab_prediccion, tab_secuencias = st.tabs([
        "🔍 Patrones (Números)", 
        "📅 Almanaque", 
        "🧠 Propuesta Inteligente",
        "🔗 Secuencia de Dígitos"
    ])

    # ==========================================
    # PESTAÑA 1: PATRONES DE NÚMEROS
    # ==========================================
    with tab_patrones:
        st.subheader(f"Análisis de Números: {titulo_sesion}")
        col_a, col_b = st.columns(2)
        with col_a:
            num_patron = st.number_input("Número Disparador:", min_value=0, max_value=99, value=40, format="%02d")
        with col_b:
            ven_patron = st.slider("Ventana (sorteos):", min_value=1, max_value=30, value=15, step=1)
        
        if st.button("🔍 Buscar", key="btn_patron", type="secondary"):
            st.session_state['buscar_patron'] = True
        
        if st.session_state.get('buscar_patron', False):
            df_res, total = analizar_siguientes(df_analisis, num_patron, ven_patron)
            if df_res is None:
                st.error(f"El número **{num_patron:02d}** no ha salido.")
            else:
                st.success(f"Basado en {total} oportunidades.")
                max_val = int(df_res['Frecuencia'].max())
                st.dataframe(df_res.head(20), column_config={"Número": st.column_config.TextColumn("Número", width="small"), "Frecuencia": st.column_config.ProgressColumn("Frecuencia", format="%d", min_value=0, max_value=max_val)}, hide_index=True)

    # ==========================================
    # PESTAÑA 2: ALMANAQUE
    # ==========================================
    with tab_almanaque:
        st.subheader(f"Almanaque: {titulo_sesion}")
        col1, col2 = st.columns(2)
        with col1:
            tipo_quincena = st.radio("Quincena:", ["1ra Quincena (1-15)", "2da Quincena (16-30)"])
        with col2:
            meses_atras = st.slider("Meses atrás:", min_value=1, max_value=12, value=3, step=1)
        
        if st.button("📊 Analizar", key="btn_almanaque", type="secondary"):
            st.session_state['analisis_almanaque'] = True
            
        if st.session_state.get('analisis_almanaque', False):
            df_filtrado, decenas, unidades = analizar_almanaque(df_analisis, tipo_quincena, meses_atras)
            if df_filtrado is None:
                st.warning("Sin datos.")
            else:
                general = decenas + unidades
                df_g = general.reset_index(name='Total')
                df_g.columns = ['Dígito', 'Total']; df_g = df_g.sort_values(by='Total', ascending=False).reset_index(drop=True)
                df_d = decenas.reset_index(name='Decenas'); df_d.columns = ['Dígito', 'Decenas']; df_d = df_d.sort_values(by='Decenas', ascending=False).reset_index(drop=True)
                df_u = unidades.reset_index(name='Unidades'); df_u.columns = ['Dígito', 'Unidades']; df_u = df_u.sort_values(by='Unidades', ascending=False).reset_index(drop=True)
                c1, c2, c3 = st.columns(3)
                with c1: st.subheader("🏆 General"); st.dataframe(df_g, hide_index=True)
                with c2: st.subheader("🔢 Decenas"); st.dataframe(df_d, hide_index=True)
                with c3: st.subheader("🔢 Unidades"); st.dataframe(df_u, hide_index=True)

    # ==========================================
    # PESTAÑA 3: PROPUESTA INTELIGENTE
    # ==========================================
    with tab_prediccion:
        st.subheader(f"Sincronización: {titulo_sesion}")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            dias_tendencia = st.number_input("Días Tendencia:", value=15, min_value=5, max_value=60)
        with col_t2:
            dias_minimo_gap = st.number_input("Gap Mínimo:", value=10, min_value=1, max_value=90)
            
        if st.button("🧠 Generar", type="primary"):
            st.session_state['generar_prediccion'] = True
            
        if st.session_state.get('generar_prediccion', False):
            df_propuesta = generar_sugerencia_tendencia(df_analisis, dias_tendencia, dias_minimo_gap)
            if df_propuesta.empty:
                st.warning("Sin candidatos.")
            else:
                st.success(f"{len(df_propuesta)} candidatos.")
                st.dataframe(df_propuesta, hide_index=True, use_container_width=True)

    # ==========================================
    # PESTAÑA 4: SECUENCIA DE DÍGITOS (ACTUALIZADA CON FECHA)
    # ==========================================
    with tab_secuencias:
        st.subheader(f"Patrones de Decenas/Unidades: {titulo_sesion}")
        st.markdown("""
        Busca una consecutividad de dígitos (ej: 5 8 3) y te dice qué dígito salió después históricamente.
        Ahora incluye **Ejemplos de Números y la Fecha** exacta.
        """)
        
        col_tipo, col_input = st.columns([1, 2])
        with col_tipo:
            tipo_digito = st.selectbox("Analizar:", ["Decena", "Unidad"])
        
        with col_input:
            secuencia_input = st.text_input(
                "Ingresa la secuencia (ej: 5 8 3 7):", 
                placeholder="Usa espacios o comas. Máx 5 dígitos."
            )
            
        if st.button("🔗 Buscar Secuencia", type="primary"):
            st.session_state['buscar_secuencia'] = True
            
        if st.session_state.get('buscar_secuencia', False) and secuencia_input:
            df_res, error = buscar_secuencia_digitos(df_analisis, secuencia_input, tipo_digito)
            
            st.markdown("---")
            if error:
                st.warning(error)
            else:
                st.success("✅ Secuencia encontrada en el historial.")
                st.dataframe(
                    df_res, 
                    column_config={
                        "Dígito Siguiente": st.column_config.NumberColumn("Dígito Siguiente", width="small"),
                        "Frecuencia": st.column_config.ProgressColumn("Frecuencia", format="%d", min_value=0, max_value=int(df_res['Frecuencia'].max())),
                        "Probabilidad (%)": st.column_config.NumberColumn("Probabilidad", format="%.2f%%"),
                        "Ejemplos Históricos (Núm + Fecha)": st.column_config.TextColumn("Ejemplos Reales", width="medium")
                    },
                    hide_index=True, use_container_width=True
                )

if __name__ == "__main__":
    main()