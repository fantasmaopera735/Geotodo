# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import traceback
import time 
from collections import defaultdict, Counter
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN DE LA RUTA_RELATIVA ---
RUTA_CSV = 'Geotodo.csv' 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Geotodo - Análisis de Sorteos",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Geotodo - Análisis de Sorteos")
st.markdown("Sistema de Análisis para los sorteos de Geotodo (Mañana, Tarde y Noche).")
st.info("ℹ️ **Importante:** Cálculos basados en PROMEDIO (Mean) SOLO para el número **FIJO**.")

# --- FUNCIÓN AUXILIAR PARA ELIMINAR ACENTOS ---
def remove_accents(input_str):
    if not isinstance(input_str, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- FUNCIÓN PARA CARGAR Y PROCESAR DATOS ---
@st.cache_resource
def cargar_datos_geotodo(_ruta_csv, debug_mode=False):
    try:
        st.info("Cargando y procesando datos históricos de Geotodo...")
        ruta_csv_absoluta = _ruta_csv
        
        if not os.path.exists(ruta_csv_absoluta):
            st.error(f"❌ Error: No se encontró el archivo de datos de Geotodo.")
            st.error(f"La aplicación buscó el archivo en la ruta: {ruta_csv_absoluta}")
            st.warning("💡 **Solución:** Asegúrate de que el archivo 'Geotodo.csv' exista en la carpeta raíz.")
            st.stop()
        
        with open(ruta_csv_absoluta, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        
        if not lines:
            st.error("El archivo CSV está vacío.")
            st.stop()
        
        header_line = lines[0].strip()
        column_names = header_line.split(';')
        
        data = []
        for line in lines[1:]:
            if line.strip():
                values = line.strip().split(';')
                if len(values) >= 5:
                    data.append(values)
        
        df_historial = pd.DataFrame(data, columns=column_names)
        
        if debug_mode:
            st.subheader("🔍 Examen de los Encabezados del CSV")
            st.write("**Lista completa de encabezados:**")
            st.code(header_line)
            st.dataframe(df_historial.head())
        
        df_historial.rename(columns={
            'Fecha': 'Fecha',
            'Tarde/Noche': 'Tipo_Sorteo', 
            'Fijo': 'Fijo',
            '1er Corrido': 'Primer_Corrido',
            '2do Corrido': 'Segundo_Corrido'
        }, inplace=True)
        
        df_historial['Fecha'] = pd.to_datetime(df_historial['Fecha'], dayfirst=True, errors='coerce')
        df_historial.dropna(subset=['Fecha'], inplace=True)
        
        st.write("Normalizando la columna 'Tipo_Sorteo' para Geotodo (M, T, N)...")
        df_historial['Tipo_Sorteo'] = df_historial['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({
            'MAÑANA': 'M', 'M': 'M', 'MANANA': 'M',
            'TARDE': 'T', 'T': 'T',
            'NOCHE': 'N', 'N': 'N'
        }).fillna('OTRO')
        st.success("Columna 'Tipo_Sorteo' normalizada (M, T, N).")
        if debug_mode:
            st.write("Valores únicos en 'Tipo_Sorteo' después de normalizar:", df_historial['Tipo_Sorteo'].unique())
        
        # --- CARGAR TODOS LOS DATOS (FIJO + CORRIDOS) ---
        df_procesado = []
        for _, row in df_historial.iterrows():
            fecha = row['Fecha']
            tipo_sorteo = row['Tipo_Sorteo']
            try:
                fijo = int(row['Fijo']) if pd.notna(row['Fijo']) else 0
                p1 = int(row['Primer_Corrido']) if pd.notna(row['Primer_Corrido']) else 0
                p2 = int(row['Segundo_Corrido']) if pd.notna(row['Segundo_Corrido']) else 0
            except ValueError:
                continue

            df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo_sorteo, 'Numero': fijo, 'Posicion': 'Fijo'})
            df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo_sorteo, 'Numero': p1, 'Posicion': '1er Corrido'})
            df_procesado.append({'Fecha': fecha, 'Tipo_Sorteo': tipo_sorteo, 'Numero': p2, 'Posicion': '2do Corrido'})
        
        df_historial = pd.DataFrame(df_procesado)
        df_historial['Numero'] = pd.to_numeric(df_historial['Numero'], errors='coerce')
        df_historial.dropna(subset=['Numero'], inplace=True)
        df_historial['Numero'] = df_historial['Numero'].astype(int)
        
        # Orden cronológico
        draw_order_map = {'M': 0, 'T': 1, 'N': 2}
        df_historial['draw_order'] = df_historial['Tipo_Sorteo'].map(draw_order_map).fillna(3)
        df_historial['sort_key'] = df_historial['Fecha'] + pd.to_timedelta(df_historial['draw_order'], unit='h')
        df_historial = df_historial.sort_values(by='sort_key').reset_index(drop=True)
        df_historial.drop(columns=['draw_order', 'sort_key'], inplace=True)
        
        st.success("¡Datos de Geotodo cargados y procesados con éxito!")
        return df_historial
    except Exception as e:
        st.error(f"Error al cargar y procesar los datos de Geotodo: {str(e)}")
        if debug_mode:
            st.error(traceback.format_exc())
        st.stop()

# --- FUNCIÓN PARA CALCULAR ESTADO ACTUAL ---
def calcular_estado_actual(gap, promedio_gap):
    if pd.isna(promedio_gap) or promedio_gap == 0:
        return "Normal"
    
    if gap <= promedio_gap:
        return "Normal"
    elif gap > (promedio_gap * 1.5):
        return "Muy Vencido"
    else: 
        return "Vencido"

# --- FUNCIÓN PARA OBTENER ESTADO COMPLETO DE NÚMEROS (SOLO FIJO) ---
def get_full_state_dataframe(df_historial, fecha_referencia):
    df_solo_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    df_historial_filtrado = df_solo_fijo[df_solo_fijo['Fecha'] < fecha_referencia].copy()
    
    if df_historial_filtrado.empty:
        return pd.DataFrame(), {}

    df_maestro = pd.DataFrame({'Numero': range(100)})
    primera_fecha_historica = df_historial_filtrado['Fecha'].min()
    
    historicos_numero = {}
    for i in range(100):
        fechas_i = df_historial_filtrado[df_historial_filtrado['Numero'] == i]['Fecha'].sort_values()
        gaps = fechas_i.diff().dt.days.dropna()
        
        if len(gaps) > 0:
            historicos_numero[i] = gaps.mean()
        else:
            historicos_numero[i] = (fecha_referencia - primera_fecha_historica).days

    df_maestro['Decena'] = df_maestro['Numero'] // 10
    df_maestro['Unidad'] = df_maestro['Numero'] % 10
    
    ultima_aparicion_num_key = df_historial_filtrado.groupby('Numero')['Fecha'].max().reindex(range(100))
    ultima_aparicion_num_key.fillna(primera_fecha_historica, inplace=True)
    gap_num = (fecha_referencia - ultima_aparicion_num_key).dt.days
    df_maestro['Salto_Numero'] = gap_num
    df_maestro['Estado_Numero'] = df_maestro.apply(lambda row: calcular_estado_actual(row['Salto_Numero'], historicos_numero[row['Numero']]), axis=1)
    df_maestro['Última Aparición (Fecha)'] = ultima_aparicion_num_key.dt.strftime('%d/%m/%Y')
    frecuencia = df_historial_filtrado['Numero'].value_counts().reindex(range(100)).fillna(0)
    df_maestro['Total_Salidas_Historico'] = frecuencia

    return df_maestro, historicos_numero

# --- FUNCIÓN PARA CLASIFICAR NÚMEROS POR TEMPERATURA ---
def crear_mapa_de_calor_numeros(df_frecuencia, top_n=30, medio_n=30):
    df_ordenado = df_frecuencia.sort_values(by='Total_Salidas_Historico', ascending=False).reset_index(drop=True).copy()
    df_ordenado['Temperatura'] = '🧊 Frío'
    df_ordenado.loc[top_n : top_n + medio_n - 1, 'Temperatura'] = '🟡 Tibio'
    df_ordenado.loc[0 : top_n - 1, 'Temperatura'] = '🔥 Caliente'
    return df_ordenado

# --- NUEVO: BACKTEST DE GRUPOS Y ESTRATEGIA ---
def realizar_backtest_predicciones(df_historial, meses_atras=6):
    """
    Simula la estrategia 'Top 5 por Grupo' y los grupos generales en el pasado
    para calcular la frecuencia de aciertos y el gap promedio.
    """
    st.info(f"🔄 Iniciando Backtesting (Simulación histórica) de los últimos {meses_atras} meses...")
    fecha_limite = df_historial['Fecha'].max() - pd.DateOffset(months=meses_atras)
    
    # Filtrar historial para el rango
    df_simulacion = df_historial[df_historial['Fecha'] >= fecha_limite].copy()
    if df_simulacion.empty:
        return {}, {}

    # --- 1. ANÁLISIS DE GRUPOS GENERALES (Simple) ---
    resultados_grupos = {'🔥 Caliente': [], '🟡 Tibio': [], '🧊 Frío': []}
    
    fechas_unicas = sorted(df_simulacion['Fecha'].unique())
    
    for fecha in fechas_unicas:
        # Datos hasta esa fecha (sin incluir el de hoy para no contaminar el cálculo de frecuencia)
        df_hasta_ayer = df_historial[df_historial['Fecha'] < fecha].copy()
        
        if df_hasta_ayer.empty: continue
            
        # Calcular temperaturas actuales para ese día
        frecuencia_actual = df_hasta_ayer['Numero'].value_counts().reset_index()
        frecuencia_actual.columns = ['Numero', 'Total_Salidas_Historico']
        df_temp = crear_mapa_de_calor_numeros(frecuencia_actual)
        
        # Obtener ganador de HOY (fecha) - CORRECCIÓN DE BUG AQUÍ
        # Puede haber múltiples ganadores si es Análisis General (M, T, N)
        ganadores_hoy = df_simulacion[df_simulacion['Fecha'] == fecha]['Numero'].values
        
        if len(ganadores_hoy) == 0: continue
        
        # Chequear temperatura de CADA ganador
        for ganador_num in ganadores_hoy:
            if ganador_num in df_temp['Numero'].values:
                temp_ganador = df_temp[df_temp['Numero'] == ganador_num]['Temperatura'].values[0]
                if temp_ganador in resultados_grupos:
                    resultados_grupos[temp_ganador].append(fecha)

    # Calcular Gaps para Grupos Generales
    stats_grupos = {}
    for grupo, fechas in resultados_grupos.items():
        if len(fechas) < 2:
            stats_grupos[grupo] = {'Aciertos': len(fechas), 'Gap Promedio': 'N/A', 'Max Gap': 'N/A'}
        else:
            gaps = pd.Series(fechas).diff().dt.days.dropna()
            stats_grupos[grupo] = {
                'Aciertos': len(fechas), 
                'Gap Promedio': round(gaps.mean(), 1),
                'Max Gap': int(gaps.max()),
                'Min Gap': int(gaps.min())
            }

    # --- 2. ANÁLISIS DE ESTRATEGIA TOP 5 (Complejo) ---
    fechas_aciertos_estrategia = []
    
    for fecha in fechas_unicas:
        # Paso 1: Simular el estado de la APP "ayer"
        df_ayer = df_historial[df_historial['Fecha'] < fecha].copy()
        
        if df_ayer.empty: continue
        
        # Paso 2: Calcular Clasificación (Hot/Tibio/Frio) de ayer
        freq_ayer = df_ayer['Numero'].value_counts().reset_index()
        freq_ayer.columns = ['Numero', 'Total_Salidas_Historico']
        df_clasif_ayer = crear_mapa_de_calor_numeros(freq_ayer)
        
        # Paso 3: Calcular Estado (Normal/Vencido/MuyVencido)
        df_estados_ayer, _ = get_full_state_dataframe(df_ayer, fecha)
        
        # Paso 4: Calcular Prioridad de Grupos
        grupos_analizar = ['🔥 Caliente', '🟡 Tibio', '🧊 Frío']
        mejor_grupo = '🟡 Tibio' 
        max_puntuacion = -1
        
        for temp in grupos_analizar:
            nums_grupo = df_clasif_ayer[df_clasif_ayer['Temperatura'] == temp]['Numero'].tolist()
            # Cruzar con estados
            nums_con_estado = df_estados_ayer[df_estados_ayer['Numero'].isin(nums_grupo)]
            vencidos = len(nums_con_estado[nums_con_estado['Estado_Numero'].isin(['Vencido', 'Muy Vencido'])])
            
            if vencidos > max_puntuacion:
                max_puntuacion = vencidos
                mejor_grupo = temp
        
        # Paso 5: Seleccionar Top 5 del Mejor Grupo
        nums_mejor_grupo = df_clasif_ayer[df_clasif_ayer['Temperatura'] == mejor_grupo]['Numero'].tolist()
        df_mejor_grupo_estado = df_estados_ayer[df_estados_ayer['Numero'].isin(nums_mejor_grupo)]
        
        estado_peso = {'Muy Vencido': 0, 'Vencido': 1, 'Normal': 2}
        df_ordenada = df_mejor_grupo_estado.copy()
        df_ordenada['peso_estado'] = df_ordenada['Estado_Numero'].map(estado_peso)
        df_ordenada = df_ordenada.sort_values(by=['peso_estado', 'Salto_Numero'], ascending=[True, False])
        
        top_5_prediccion = df_ordenada.head(5)['Numero'].tolist()
        
        # Paso 6: Comprobar acierto HOY (puede ser M, T, N)
        ganadores_hoy = df_simulacion[df_simulacion['Fecha'] == fecha]['Numero'].values
        
        # Si ALGUNO de los ganadores de hoy está en la predicción, cuenta como acierto
        if any(g in top_5_prediccion for g in ganadores_hoy):
            fechas_aciertos_estrategia.append(fecha)
    
    # Calcular Gaps para Estrategia
    stats_estrategia = {
        'Total Predicciones': len(fechas_unicas),
        'Aciertos': len(fechas_aciertos_estrategia),
        'Tasa Acierto (%)': round(len(fechas_aciertos_estrategia) / len(fechas_unicas) * 100, 1) if fechas_unicas else 0
    }
    
    if len(fechas_aciertos_estrategia) < 2:
        stats_estrategia['Gap Promedio'] = 'N/A'
        stats_estrategia['Max Gap'] = 'N/A'
    else:
        gaps_estrategia = pd.Series(fechas_aciertos_estrategia).diff().dt.days.dropna()
        stats_estrategia['Gap Promedio'] = round(gaps_estrategia.mean(), 1)
        stats_estrategia['Max Gap'] = int(gaps_estrategia.max())
        stats_estrategia['Min Gap'] = int(gaps_estrategia.min())

    return stats_grupos, stats_estrategia

# --- NUEVA FUNCIÓN: ANÁLISIS DE REBOTE TEMPORAL (EVENTO) ---
def analizar_evento_rebote_temporal(df_completo, fecha_referencia, meses_atras=6):
    """
    Analiza el fenómeno 'Rebote' como un EVENTO TEMPORAL.
    Objetivo: Saber CUÁNTO tiempo ha pasado desde el último rebote y si estamos cerca del promedio.
    """
    fecha_limite = fecha_referencia - pd.DateOffset(months=meses_atras)
    
    df_historial_filtrado = df_completo[df_completo['Fecha'] >= fecha_limite].copy()
    df_fijos = df_historial_filtrado[df_historial_filtrado['Posicion'] == 'Fijo'].sort_values('Fecha')
    df_corridos = df_historial_filtrado[df_historial_filtrado['Posicion'].isin(['1er Corrido', '2do Corrido'])].sort_values('Fecha')
    
    fechas_con_evento_rebote = []
    fechas_unicas_sorteos = df_fijos['Fecha'].unique()
    
    for fecha in fechas_unicas_sorteos:
        fijos_hoy = df_fijos[df_fijos['Fecha'] == fecha]['Numero'].tolist()
        ventana_inicio = fecha - pd.Timedelta(days=1)
        ventana_fin = fecha
        corridos_ayer = df_corridos[
            (df_corridos['Fecha'] >= ventana_inicio) & 
            (df_corridos['Fecha'] < ventana_fin)
        ]['Numero'].tolist()
        
        if any(num in corridos_ayer for num in fijos_hoy):
            fechas_con_evento_rebote.append(fecha)
            
    if len(fechas_con_evento_rebote) < 2:
        return {
            'estado': 'Insuficientes Datos',
            'ultimo_evento': fechas_con_evento_rebote[-1] if fechas_con_evento_rebote else None,
            'dias_sin_evento': None,
            'promedio_dias': None,
            'detalles': fechas_con_evento_rebote
        }
        
    df_eventos = pd.DataFrame({'Fecha': fechas_con_evento_rebote}).sort_values('Fecha')
    gaps_eventos = df_eventos['Fecha'].diff().dt.days.dropna()
    
    promedio_dias_entre_rebotes = gaps_eventos.mean()
    ultimo_evento_fecha = df_eventos['Fecha'].max()
    dias_desde_ultimo = (fecha_referencia - ultimo_evento_fecha).days
    
    if dias_desde_ultimo > (promedio_dias_entre_rebotes * 1.2):
        estado_evento = "🔴 EVENTO MUY VENCIDO (Alta probabilidad de que ocurra hoy)"
    elif dias_desde_ultimo > promedio_dias_entre_rebotes:
        estado_evento = "🟡 EVENTO VENCIDO (Zona de rebote)"
    else:
        estado_evento = "🟢 EVENTO NORMAL (Dentro de promedio)"

    return {
        'estado': estado_evento,
        'ultimo_evento': ultimo_evento_fecha,
        'dias_sin_evento': dias_desde_ultimo,
        'promedio_dias': round(promedio_dias_entre_rebotes, 1),
        'gaps_historicos': gaps_eventos.tolist(),
        'detalles': fechas_con_evento_rebote
    }

# --- FUNCIÓN PARA ANÁLISIS DE OPORTUNIDAD POR DÍGITO ---
def analizar_oportunidad_por_digito(df_historial, df_estados_completos, historicos_numero, modo_temperatura, fecha_inicio_rango, fecha_fin_rango):
    df_solo_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    
    if modo_temperatura == "Histórico Completo":
        df_temperatura = df_solo_fijo.copy()
    else:
        end_of_day = fecha_fin_rango + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_temperatura = df_solo_fijo[(df_solo_fijo['Fecha'] >= fecha_inicio_rango) & (df_solo_fijo['Fecha'] <= end_of_day)].copy()
        if df_temperatura.empty:
            df_temperatura = df_solo_fijo.copy()

    contador_decenas = Counter()
    contador_unidades = Counter()
    for num in df_temperatura['Numero']:
        contador_decenas[num // 10] += 1
        contador_unidades[num % 10] += 1

    df_frecuencia_decenas = pd.DataFrame.from_dict(contador_decenas, orient='index', columns=['Frecuencia Total']).reset_index()
    df_frecuencia_decenas.rename(columns={'index': 'Dígito'}, inplace=True)
    df_frecuencia_unidades = pd.DataFrame.from_dict(contador_unidades, orient='index', columns=['Frecuencia Total']).reset_index()
    df_frecuencia_unidades.rename(columns={'index': 'Dígito'}, inplace=True)

    df_frecuencia_decenas = df_frecuencia_decenas.sort_values(by='Frecuencia Total', ascending=False).reset_index(drop=True)
    df_frecuencia_decenas['Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_decenas) >= 3: df_frecuencia_decenas.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df_frecuencia_decenas) >= 6: df_frecuencia_decenas.loc[3:5, 'Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_decenas) >= 7: df_frecuencia_decenas.loc[6:9, 'Temperatura'] = '🧊 Frío'
    
    df_frecuencia_unidades = df_frecuencia_unidades.sort_values(by='Frecuencia Total', ascending=False).reset_index(drop=True)
    df_frecuencia_unidades['Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_unidades) >= 3: df_frecuencia_unidades.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df_frecuencia_unidades) >= 6: df_frecuencia_unidades.loc[3:5, 'Temperatura'] = '🟡 Tibio'
    if len(df_frecuencia_unidades) >= 7: df_frecuencia_unidades.loc[6:9, 'Temperatura'] = '🧊 Frío'
    
    mapa_temperatura_decenas = pd.Series(df_frecuencia_decenas.Temperatura.values, index=df_frecuencia_decenas.Dígito).to_dict()
    mapa_temperatura_unidades = pd.Series(df_frecuencia_unidades.Temperatura.values, index=df_frecuencia_unidades.Dígito).to_dict()
    
    puntuacion_temperatura_map = {'🔥 Caliente': 30, '🟡 Tibio': 20, '🧊 Frío': 10}
    resultados_decenas = []
    resultados_unidades = []
    
    for i in range(10):
        numeros_con_decena = df_estados_completos[df_estados_completos['Decena'] == i]
        numeros_con_unidad = df_estados_completos[df_estados_completos['Unidad'] == i]
        
        estado_decena = numeros_con_decena['Estado_Numero'].iloc[0] if not numeros_con_decena.empty else 'Normal'
        estado_unidad = numeros_con_unidad['Estado_Numero'].iloc[0] if not numeros_con_unidad.empty else 'Normal'

        puntuacion_base_decena = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[estado_decena]
        puntuacion_base_unidad = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}[estado_unidad]

        if estado_decena == 'Normal':
            gaps_decena = [historicos_numero.get(num, 0) for num in range(100) if num // 10 == i]
            promedio = np.mean(gaps_decena) if gaps_decena else 1
            gaps_actuales = numeros_con_decena['Salto_Numero'].tolist()
            gap_actual = np.mean(gaps_actuales) if gaps_actuales else 0
            puntuacion_proactiva_decena = min(49, (gap_actual / promedio * 50) if promedio > 0 else 0)
        else:
            puntuacion_proactiva_decena = 0

        if estado_unidad == 'Normal':
            gaps_unidad = [historicos_numero.get(num, 0) for num in range(100) if num % 10 == i]
            promedio = np.mean(gaps_unidad) if gaps_unidad else 1
            gaps_actuales = numeros_con_unidad['Salto_Numero'].tolist()
            gap_actual = np.mean(gaps_actuales) if gaps_actuales else 0
            puntuacion_proactiva_unidad = min(49, (gap_actual / promedio * 50) if promedio > 0 else 0)
        else:
            puntuacion_proactiva_unidad = 0

        temperatura_decena = mapa_temperatura_decenas.get(i, '🟡 Tibio')
        temperatura_unidad = mapa_temperatura_unidades.get(i, '🟡 Tibio')
        puntuacion_temp_decena = puntuacion_temperatura_map.get(temperatura_decena, 20)
        puntuacion_temp_unidad = puntuacion_temperatura_map.get(temperatura_unidad, 20)

        puntuacion_total_decena = puntuacion_base_decena + puntuacion_proactiva_decena + puntuacion_temp_decena
        puntuacion_total_unidad = puntuacion_base_unidad + puntuacion_proactiva_unidad + puntuacion_temp_unidad

        resultados_decenas.append({'Dígito': i, 'Rol': 'Decena', 'Temperatura': temperatura_decena, 'Estado': estado_decena, 'Puntuación Base': puntuacion_base_decena, 'Puntuación Proactiva': round(puntuacion_proactiva_decena, 1), 'Puntuación Temperatura': puntuacion_temp_decena, 'Puntuación Total': round(puntuacion_total_decena, 1)})
        resultados_unidades.append({'Dígito': i, 'Rol': 'Unidad', 'Temperatura': temperatura_unidad, 'Estado': estado_unidad, 'Puntuación Base': puntuacion_base_unidad, 'Puntuación Proactiva': round(puntuacion_proactiva_unidad, 1), 'Puntuación Temperatura': puntuacion_temp_unidad, 'Puntuación Total': round(puntuacion_total_unidad, 1)})

    df_oportunidad_decenas = pd.DataFrame(resultados_decenas)
    df_oportunidad_unidades = pd.DataFrame(resultados_unidades)

    return df_oportunidad_decenas, df_oportunidad_unidades, mapa_temperatura_decenas, mapa_temperatura_unidades

# --- FUNCIÓN: LISTADO POR DECENAS Y ESTADO ---
def mostrar_listado_decenas_detalle(df_estados_completos):
    st.subheader("📋 Listado Detallado de Números por Decenas y su Estado")
    st.markdown("A continuación se muestran los números de 00 a 99 agrupados por su Decena, junto con su estado actual individual.")
    
    cols = st.columns(5) 
    
    for i in range(10):
        with cols[i % 5]:
            st.markdown(f"### Decena {i}")
            numeros_decena = df_estados_completos[df_estados_completos['Decena'] == i].sort_values('Numero')
            
            if not numeros_decena.empty:
                for _, row in numeros_decena.iterrows():
                    num_str = f"{row['Numero']:02d}"
                    estado = row['Estado_Numero']
                    icono = "⚪"
                    if estado == "Vencido": icono = "🟡"
                    elif estado == "Muy Vencido": icono = "🔴"
                    st.markdown(f"<small>{icono} <b>{num_str}</b>: {estado}</small>", unsafe_allow_html=True)
            else:
                st.write("Sin datos")

# --- FUNCIÓN PARA BUSCAR PATRONES ---
def buscar_patrones_secuenciales(df_historial, max_longitud=3, nombre_sesion="General"):
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    if df_fijo.empty:
        return {}
    
    secuencia = df_fijo['Numero'].tolist()
    if len(df_fijo) <= max_longitud:
        return {}
        
    patrones = {}
    for longitud in range(2, max_longitud + 1):
        for i in range(len(secuencia) - longitud):
            patron = tuple(secuencia[i:i+longitud])
            siguiente = secuencia[i+longitud] if i+longitud < len(secuencia) else None
            if siguiente is not None:
                if patron not in patrones: patrones[patron] = {}
                if siguiente not in patrones[patron]: patrones[patron][siguiente] = 0
                patrones[patron][siguiente] += 1
                
    patrones_ordenados = {}
    for patron, siguientes in patrones.items():
        siguientes_ordenados = sorted(siguientes.items(), key=lambda x: x[1], reverse=True)
        patrones_ordenados[patron] = siguientes_ordenados
    return patrones_ordenados

# --- FUNCIÓN PARA COMPORTAMIENTO DOBLE NORMAL ---
def analizar_comportamiento_doble_normal(df_historial):
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy() 
    if df_fijo.empty: return pd.DataFrame()

    comportamiento = {'Normal': 0, 'Vencido': 0, 'Muy Vencido': 0}
    total_doble_normal = 0
    historial_fechas = {num: df_fijo[df_fijo['Numero'] == num]['Fecha'].sort_values() for num in range(100)}
    
    for i, row in df_fijo.iterrows():
        fecha_actual, numero_actual = row['Fecha'], row['Numero']
        decena, unidad = numero_actual // 10, numero_actual % 10
        es_doble_normal = True
        
        for digito in [decena, unidad]:
            fechas_digito = [f for f in historial_fechas[digito] if f < fecha_actual]
            if len(fechas_digito) == 0: es_doble_normal = False; break
            gaps = pd.Series(fechas_digito).diff().dt.days.dropna()
            if len(gaps) == 0: es_doble_normal = False; break
            promedio_gap = gaps.mean()
            gap_actual = (fecha_actual - fechas_digito[-1]).days
            if calcular_estado_actual(gap_actual, promedio_gap) != 'Normal':
                es_doble_normal = False; break
        
        if es_doble_normal:
            total_doble_normal += 1
            fechas_numero = [f for f in historial_fechas[numero_actual] if f < fecha_actual]
            if not fechas_numero: continue
            gaps_num = pd.Series(fechas_numero).diff().dt.days.dropna()
            if len(gaps_num) == 0: continue
            promedio_gap_num = gaps_num.mean()
            gap_actual_num = (fecha_actual - fechas_numero[-1]).days
            estado_num = calcular_estado_actual(gap_actual_num, promedio_gap_num)
            comportamiento[estado_num] += 1
            
    if total_doble_normal == 0:
        return pd.DataFrame()

    df_comportamiento = pd.DataFrame(list(comportamiento.items()), columns=['Estado del Número', 'Cantidad'])
    df_comportamiento['Porcentaje'] = (df_comportamiento['Cantidad'] / total_doble_normal * 100).round(1)
    return df_comportamiento

# --- FUNCIÓN TABLA HISTÓRICO VISUAL ---
def crear_tabla_historico_visual_fijo(df_historial, num_ultimos=30):
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    if df_fijo.empty:
        return pd.DataFrame()

    fecha_max = df_fijo['Fecha'].max()
    fecha_inicio_4s = fecha_max - pd.Timedelta(weeks=4)
    df_analisis_4s = df_fijo[df_fijo['Fecha'] >= fecha_inicio_4s].copy()
    
    if df_analisis_4s.empty:
        return pd.DataFrame()

    ultimas_fechas = df_fijo['Fecha'].unique()[-num_ultimos:]
    historial_visual = []
    
    for fecha in sorted(ultimas_fechas, reverse=True):
        for sesion_key, sesion_nombre in [('N', 'Noche'), ('T', 'Tarde'), ('M', 'Mañana')]:
            resultado = df_fijo[(df_fijo['Fecha'] == fecha) & (df_fijo['Tipo_Sorteo'] == sesion_key)]
            if not resultado.empty:
                numero = resultado.iloc[0]['Numero']
                df_hasta_fecha = df_analisis_4s[df_analisis_4s['Fecha'] < fecha]
                
                def get_estado_en_fecha(num, df_hist, fecha_limite):
                    df_num = df_hist[df_hist['Numero'] == num]['Fecha'].sort_values()
                    if len(df_num) == 0: return 'Normal', 0, 0
                    gaps = df_num.diff().dt.days.dropna()
                    if len(gaps) == 0: return 'Normal', 0, 0
                    promedio = gaps.mean() 
                    ultima_fecha = df_num.max()
                    gap = (fecha_limite - ultima_fecha).days
                    estado = calcular_estado_actual(gap, promedio)
                    return estado, gap, promedio
                
                estado_dec, _, _ = get_estado_en_fecha(numero, df_hasta_fecha, fecha)
                estado_uni, _, _ = get_estado_en_fecha(numero, df_hasta_fecha, fecha)
                es_doble_normal = (estado_dec == 'Normal' and estado_uni == 'Normal')
                
                contexto_hist = df_hasta_fecha if df_analisis_4s.empty else df_fijo.copy()
                df_num_hist = contexto_hist[contexto_hist['Numero'] == numero]
                
                if df_num_hist.empty:
                    temp = 'N/A'
                    estado_num = 'N/A'
                else:
                    df_freq_total = contexto_hist['Numero'].value_counts().reset_index()
                    df_freq_total.columns = ['Numero', 'Total_Salidas_Historico']
                    df_freq = crear_mapa_de_calor_numeros(df_freq_total)
                    row = df_freq[df_freq['Numero'] == numero]
                    if not row.empty:
                        temp = row['Temperatura'].iloc[0]
                    else:
                        temp = 'N/A'
                    
                    gaps_num = df_num_hist['Fecha'].diff().dt.days.dropna()
                    if len(gaps_num) == 0:
                        estado_num = 'N/A'
                    else:
                        promedio_gap_num = gaps_num.mean()
                        ultima_fecha_num = df_num_hist['Fecha'].max()
                        gap_num = (fecha - ultima_fecha_num).days
                        estado_num = calcular_estado_actual(gap_num, promedio_gap_num)

                historial_visual.append({
                    'Fecha': fecha.strftime('%d/%m/%Y'),
                    'Sesión': sesion_nombre,
                    'Fijo': f"{numero:02d}",
                    'Es Doble Normal': es_doble_normal,
                    'Temperatura': temp,
                    'Estado': estado_num
                })

    return pd.DataFrame(historial_visual)

# --- FUNCIÓN ESTRATEGIA TENDENCIA ---
def generar_estrategia_tendencia(df_historial, fecha_referencia):
    df_fijo = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    df_fijo = df_fijo[df_fijo['Fecha'] < fecha_referencia].copy()
    if df_fijo.empty: return pd.DataFrame(), [], [], []

    estados_digito = {d: {'estado': 'Normal', 'gap': 0, 'promedio': 0} for d in range(10)}
    for d in range(10):
        numeros_con_digito = [n for n in range(100) if n // 10 == d or n % 10 == d]
        fechas_digito = df_fijo[df_fijo['Numero'].isin(numeros_con_digito)]['Fecha'].sort_values()
        gaps = fechas_digito.diff().dt.days.dropna()
        if len(gaps) > 0:
            promedio_gap = gaps.mean()
            ultima_fecha = fechas_digito.max()
            gap_actual = (fecha_referencia - ultima_fecha).days
            estado = calcular_estado_actual(gap_actual, promedio_gap)
            estados_digito[d] = {'estado': estado, 'gap': gap_actual, 'promedio': promedio_gap}

    estados_numero = {n: {'estado': 'Normal', 'gap': 0, 'promedio': 0} for n in range(100)}
    for n in range(100):
        fechas_num = df_fijo[df_fijo['Numero'] == n]['Fecha'].sort_values()
        gaps = fechas_num.diff().dt.days.dropna()
        if len(gaps) > 0:
            promedio_gap = gaps.mean()
            ultima_fecha_num = fechas_num.max()
            gap_actual = (fecha_referencia - ultima_fecha_num).days
            estado = calcular_estado_actual(gap_actual, promedio_gap)
            estados_numero[n] = {'estado': estado, 'gap': gap_actual, 'promedio': promedio_gap}

    todos_doble_normal = []
    for num in range(100):
        decena, unidad = num // 10, num % 10
        if (estados_digito[decena]['estado'] == 'Normal' and 
            estados_digito[unidad]['estado'] == 'Normal' and
            estados_numero[num]['estado'] == 'Normal'):
            todos_doble_normal.append(num)

    candidatos_vencidos = []
    for num in range(100):
        decena, unidad = num // 10, num % 10
        if (estados_digito[decena]['estado'] == 'Normal' and 
            estados_digito[unidad]['estado'] == 'Normal' and
            estados_numero[num]['estado'] in ['Vencido', 'Muy Vencido']):
            candidatos_vencidos.append(num)
    
    candidatos_normales = []
    for num in range(100):
        decena, unidad = num // 10, num % 10
        if (estados_digito[decena]['estado'] == 'Normal' and 
            estados_digito[unidad]['estado'] == 'Normal' and
            estados_numero[num]['estado'] == 'Normal'):
            
            fechas_num = df_fijo[df_fijo['Numero'] == num]['Fecha'].sort_values().tolist()
            if len(fechas_num) < 56: continue
                
            gaps_semanales = []
            for i in range(len(fechas_num) - 1, 0, -7):
                if i - 7 < 0: break
                gap_semanal = (fechas_num[i] - fechas_num[i-7]).days
                if gap_semanal > 0:
                    gaps_semanales.append(gap_semanal)
            
            if gaps_semanales:
                gap_promedio_semanal = np.mean(gaps_semanales)
                gap_actual_semanal = (fecha_referencia - fechas_num[-1]).days
                if gap_actual_semanal < gap_promedio_semanal:
                    candidatos_normales.append((num, gap_actual_semanal, gap_promedio_semanal))
    
    candidatos_normales.sort(key=lambda x: x[1])
    candidatos_normales = [num for num, _, _ in candidatos_normales[:10]]
    
    df_completa = pd.DataFrame({'Número': [f"{n:02d}" for n in todos_doble_normal]})
    candidatos_finales = sorted(list(set(candidatos_vencidos + candidatos_normales)))
    
    return df_completa, candidatos_finales, sorted(candidatos_vencidos), sorted(candidatos_normales)

# --- FUNCIÓN PRINCIPAL ---
def main():
    st.sidebar.header("⚙️ Opciones de Análisis - Geotodo")
    
    # --- SECCIÓN AÑADIR SORTEO ---
    with st.sidebar.expander("📝 Agregar Nuevo Sorteo (Actualizar CSV)", expanded=False):
        st.caption("Actualiza los resultados rápidamente.")
        
        fecha_nueva = st.date_input("Fecha del sorteo:", value=datetime.now().date(), format="DD/MM/YYYY")
        sesion = st.radio("Sesión:", ["Mañana (M)", "Tarde (T)", "Noche (N)"], horizontal=True)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fijo = st.number_input("Fijo", min_value=0, max_value=99, value=0, format="%02d")
        with col_b:
            p1 = st.number_input("1er Corr.", min_value=0, max_value=99, value=0, format="%02d")
        with col_c:
            p2 = st.number_input("2do Corr.", min_value=0, max_value=99, value=0, format="%02d")
        
        if st.button("💾 Guardar Sorteo", type="primary"):
            sesion_code = ""
            if "Mañana" in sesion: sesion_code = "M"
            elif "Tarde" in sesion: sesion_code = "T"
            else: sesion_code = "N"
            
            fecha_str = fecha_nueva.strftime('%d/%m/%Y')
            linea_nueva = f"{fecha_str};{sesion_code};{fijo};{p1};{p2}\n"
            
            try:
                carpeta_csv = os.path.dirname(RUTA_CSV)
                if carpeta_csv and not os.path.exists(carpeta_csv):
                    os.makedirs(carpeta_csv)
                
                with open(RUTA_CSV, 'a', encoding='latin-1') as f:
                    f.write(linea_nueva)
                
                st.success("✅ ¡Sorteo guardado!")
                st.info("Actualizando gráficos...")
                
                st.cache_resource.clear()
                time.sleep(1.5)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error al guardar: {str(e)}")

    st.sidebar.markdown("---")
    
    # --- CONFIGURACIÓN ANÁLISIS REBOTE ---
    st.sidebar.subheader("⏳ Configuración Análisis Evento Rebote")
    meses_rebote = st.sidebar.selectbox("Meses de historia para calcular ciclos de Rebote:", [3, 6, 12], index=0)

    # --- CONFIGURACIÓN BACKTEST ---
    st.sidebar.subheader("📈 Configuración Backtesting (Predicciones)")
    realizar_backtest = st.sidebar.checkbox("Activar Auditoría de Eficacia (Backtesting)", value=False)
    meses_backtest = st.sidebar.selectbox("Rango de tiempo para simulación:", [3, 6, 12], index=1)

    st.sidebar.markdown("---")
    debug_mode = st.sidebar.checkbox("🔍 Activar Modo Diagnóstico (CSV)", value=False)
    
    st.sidebar.subheader("📊 Modo de Análisis de Datos")
    modo_sorteo = st.sidebar.radio(
        "Selecciona el conjunto de datos a analizar:",
        ["Análisis General (Todos los sorteos)", "Análisis por Sesión: Mañana (M)", "Analisis por Sesión: Tarde (T)", "Análisis por Sesión: Noche (N)"]
    )
    
    modo_analisis = st.sidebar.radio(
        "Modo de Análisis Principal:",
        ["Análisis Actual (usando fecha de hoy)", "Análisis Personalizado"]
    )

    if modo_analisis == "Análisis Personalizado":
        fecha_referencia = st.sidebar.date_input("Selecciona la fecha de referencia:", value=datetime.now().date(), format="DD/MM/YYYY")
        fecha_referencia = pd.to_datetime(fecha_referencia).tz_localize(None)
    else:
        fecha_referencia = pd.Timestamp.now(tz=None)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌡️ Modo de Temperatura de Dígitos (Almanaque)")
    modo_temperatura = st.sidebar.radio(
        "Selecciona modo para calcular la temperatura:",
        ["Histórico Completo", "Personalizado por Rango"]
    )
    
    fecha_inicio_rango, fecha_fin_rango = None, None
    if modo_temperatura == "Personalizado por Rango":
        st.sidebar.markdown("**Selecciona el rango de fechas:**")
        fecha_inicio_rango = st.sidebar.date_input("Fecha de Inicio:", value=fecha_referencia - pd.Timedelta(days=30), format="DD/MM/YYYY")
        fecha_fin_rango = st.sidebar.date_input("Fecha de Fin:", value=fecha_referencia - pd.Timedelta(days=1), format="DD/MM/YYYY")
        if fecha_inicio_rango > fecha_fin_rango:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Forzar Recarga de Datos"):
        st.cache_resource.clear()
        st.sidebar.success("¡Cache limpio! Recargando...")
        st.rerun()

    df_historial_completo = cargar_datos_geotodo(RUTA_CSV, debug_mode)

    if df_historial_completo is not None:
        # --- FILTRADO CLAVE ---
        if "Mañana" in modo_sorteo:
            df_analisis = df_historial_completo[(df_historial_completo['Tipo_Sorteo'] == 'M') & (df_historial_completo['Posicion'] == 'Fijo')].copy()
            titulo_app = f"Análisis de la Sesión: Mañana (M)"
        elif "Tarde" in modo_sorteo:
            df_analisis = df_historial_completo[(df_historial_completo['Tipo_Sorteo'] == 'T') & (df_historial_completo['Posicion'] == 'Fijo')].copy()
            titulo_app = f"Análisis de la Sesión: Tarde (T)"
        elif "Noche" in modo_sorteo:
            df_analisis = df_historial_completo[(df_historial_completo['Tipo_Sorteo'] == 'N') & (df_historial_completo['Posicion'] == 'Fijo')].copy()
            titulo_app = f"Análisis de la Sesión: Noche (N)"
        else: 
            df_analisis = df_historial_completo[df_historial_completo['Posicion'] == 'Fijo'].copy()
            titulo_app = "Análisis General de Sorteos"
        
        st.title(f"🎲 {titulo_app}")
        
        if df_analisis.empty:
            st.warning(f"No hay datos para la sesión seleccionada ({modo_sorteo}).")
            st.stop()

        # --- NUEVA SECCIÓN: BACKTESTING ---
        if realizar_backtest:
            st.markdown("---")
            st.header("📈 Auditoría de Predicciones (Backtesting)")
            st.caption(f"Simulando la estrategia 'Top 5 por Prioridad' y los grupos generales en los últimos {meses_backtest} meses.")
            
            with st.spinner("Realizando simulación histórica... esto puede tomar unos segundos."):
                stats_grupos, stats_estrategia = realizar_backtest_predicciones(df_analisis, meses_atras=meses_backtest)
            
            # Mostrar resultados de Grupos Generales
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("🔥 Grupo Caliente (Gap Promedio)", stats_grupos['🔥 Caliente'].get('Gap Promedio', 'N/A'))
            with col_b2:
                st.metric("🟡 Grupo Tibio (Gap Promedio)", stats_grupos['🟡 Tibio'].get('Gap Promedio', 'N/A'))
            with col_b3:
                st.metric("🧊 Grupo Frío (Gap Promedio)", stats_grupos['🧊 Frío'].get('Gap Promedio', 'N/A'))
            
            st.markdown("---")
            # Mostrar resultados de Estrategia
            st.subheader("🎯 Resultados de la Estrategia 'Top 5 por Grupo'")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Tasa de Acierto", f"{stats_estrategia.get('Tasa Acierto (%)', 0)}%")
            with col_s2:
                st.metric("Días Promedio entre Aciertos", stats_estrategia.get('Gap Promedio', 'N/A'))
            
            st.info(f"Interpretación: En promedio, la estrategia acierta cada **{stats_estrategia.get('Gap Promedio', 'N/A')} días**.")
        
        st.sidebar.markdown("---")
        if 'M' in df_analisis['Tipo_Sorteo'].unique():
            ultimo_sorteo_M = df_analisis[df_analisis['Tipo_Sorteo'] == 'M'].iloc[-1]
            st.sidebar.info(f"Último sorteo **Mañana**: {ultimo_sorteo_M['Fecha'].strftime('%d/%m/%Y')} (Fijo: {ultimo_sorteo_M['Numero']})")
        if 'T' in df_analisis['Tipo_Sorteo'].unique():
            ultimo_sorteo_T = df_analisis[df_analisis['Tipo_Sorteo'] == 'T'].iloc[-1]
            st.sidebar.info(f"Último sorteo **Tarde**: {ultimo_sorteo_T['Fecha'].strftime('%d/%m/%Y')} (Fijo: {ultimo_sorteo_T['Numero']})")
        if 'N' in df_analisis['Tipo_Sorteo'].unique():
            ultimo_sorteo_N = df_analisis[df_analisis['Tipo_Sorteo'] == 'N'].iloc[-1]
            st.sidebar.info(f"Último sorteo **Noche**: {ultimo_sorteo_N['Fecha'].strftime('%d/%m/%Y')} (Fijo: {ultimo_sorteo_N['Numero']})")
        
        frecuencia_numeros_historica_general = df_historial_completo[df_historial_completo['Posicion'] == 'Fijo']['Numero'].value_counts().reset_index()
        frecuencia_numeros_historica_general.columns = ['Numero', 'Total_Salidas_Historico']
        df_clasificacion_general = crear_mapa_de_calor_numeros(frecuencia_numeros_historica_general)
        
        df_estados_completos, historicos_numero = get_full_state_dataframe(df_analisis, fecha_referencia)
        
        if df_estados_completos.empty:
            st.error("No se pudo calcular el estado de los números para la fecha de referencia.")
            st.stop()

        frecuencia_numeros_historica = df_analisis['Numero'].value_counts().reset_index()
        frecuencia_numeros_historica.columns = ['Numero', 'Total_Salidas_Historico']
        df_clasificacion_actual = crear_mapa_de_calor_numeros(frecuencia_numeros_historica)
        
        fecha_inicio_rango_safe = pd.to_datetime(fecha_inicio_rango).tz_localize(None) if fecha_inicio_rango else None
        fecha_fin_rango_safe = pd.to_datetime(fecha_fin_rango).tz_localize(None) if fecha_fin_rango else None
        
        df_oportunidad_decenas, df_oportunidad_unidades, mapa_temp_decenas, mapa_temp_unidades = analizar_oportunidad_por_digito(
            df_analisis, df_estados_completos, historicos_numero, 
            modo_temperatura, fecha_inicio_rango_safe, fecha_fin_rango_safe
        )
        
        # --- NUEVA SECCIÓN: ANÁLISIS DE REBOTE TEMPORAL ---
        st.markdown("---")
        st.header("⏳ Análisis de Evento 'Rebote' (Ciclos Temporales)")
        st.markdown("Este análisis mide **cuánto tiempo** pasa normalmente entre un evento de rebote (Corridoayer = Fijohoy) y el siguiente.")
        
        datos_rebote = analizar_evento_rebote_temporal(df_historial_completo, fecha_referencia, meses_rebote)
        
        col_evento_1, col_evento_2, col_evento_3 = st.columns(3)
        with col_evento_1:
            st.metric("Estado del Evento Hoy", datos_rebote['estado'])
        with col_evento_2:
            if datos_rebote['dias_sin_evento'] is not None:
                st.metric("Días desde el último Rebote", datos_rebote['dias_sin_evento'])
            else:
                st.metric("Días desde el último Rebote", "N/A")
        with col_evento_3:
            if datos_rebote['promedio_dias'] is not None:
                st.metric("Promedio Histórico (Ciclo)", f"{datos_rebote['promedio_dias']} días")
            else:
                st.metric("Promedio Histórico", "N/A")

        if datos_rebote['estado'] != 'Insuficientes Datos':
            st.info(f"📝 **Interpretación:** Históricamente, ocurre un rebote cada {datos_rebote['promedio_dias']} días en promedio. Llevamos {datos_rebote['dias_sin_evento']} días sin uno.")
        else:
            st.warning("No hay suficientes datos históricos en el rango seleccionado para calcular el ciclo de rebote.")

        # --- SECCIÓN: LISTADO POR DECENAS ---
        st.markdown("---")
        mostrar_listado_decenas_detalle(df_estados_completos)

        # --- SECCIÓN 1: NÚMEROS CON OPORTUNIDAD ---
        st.markdown("---")
        st.header("🎯 Números con Oportunidad (Debidos) por Grupo")
        st.markdown("Intersección de los números de cada grupo (Calientes, Tibios, Fríos) con los que están en estado 'Vencido' o 'Muy Vencido'.")

        oportunidades_por_grupo = {}
        grupos_analizar = ['🔥 Caliente', '🟡 Tibio', '🧊 Frío']
        for temp in grupos_analizar:
            numeros_grupo_df = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == temp]
            con_estado = numeros_grupo_df.merge(df_estados_completos[['Numero', 'Estado_Numero']], on='Numero')
            con_oportunidad = con_estado[con_estado['Estado_Numero'].isin(['Vencido', 'Muy Vencido'])]
            oportunidades_por_grupo[temp] = con_oportunidad
        
        tabs = st.tabs(grupos_analizar)
        for i, temp in enumerate(grupos_analizar):
            with tabs[i]:
                df_oportunidad_grupo = oportunidades_por_grupo[temp]
                st.subheader(f"Análisis del Grupo {temp}")
                if df_oportunidad_grupo.empty:
                    st.warning(f"Actualmente, ninguno de los números del grupo '{temp}' se encuentra en estado de 'Oportunidad'.")
                else:
                    st.success(f"Se encontraron {len(df_oportunidad_grupo)} números con 'Oportunidad' en el grupo '{temp}'.")
                    st.dataframe(df_oportunidad_grupo[['Numero', 'Total_Salidas_Historico', 'Estado_Numero']], width='stretch', hide_index=True)

        # --- SECCIÓN 2: CLASIFICACIÓN GENERAL ---
        st.markdown("---")
        st.header(f"🌡️ Clasificación de Números (Basada en {modo_sorteo})")
        
        col_cal, col_tib, col_fri = st.columns(3)
        with col_cal:
            st.metric("🔥 Calientes (Top 30)", f"{len(df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🔥 Caliente'])} números")
            calientes_lista = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🔥 Caliente']['Numero'].tolist()
            st.write(", ".join(map(str, calientes_lista)))
        with col_tib:
            st.metric("🟡 Tibios (Siguientes 30)", f"{len(df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🟡 Tibio'])} números")
            tibios_lista = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🟡 Tibio']['Numero'].tolist()
            st.write(", ".join(map(str, tibios_lista)))
        with col_fri:
            st.metric("🧊 Fríos (Últimos 40)", f"{len(df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🧊 Frío'])} números")
            frios_lista = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == '🧊 Frío']['Numero'].tolist()
            st.write(", ".join(map(str, frios_lista)))
        
        # --- SECCIÓN 3: ESTADO DE LOS GRUPOS ---
        st.markdown("---")
        st.header("📊 Estado de los Grupos por Temperatura (Rendimiento Reciente)")

        def calcular_estado_grupo_corregido(df_historial, df_clasificacion_actual, fecha_referencia, dias_recientes=7):
            resultados = {}
            df_historial_filtrado = df_historial[df_historial['Fecha'] < fecha_referencia].copy()
            total_dias_historia = (df_historial_filtrado['Fecha'].max() - df_historial_filtrado['Fecha'].min()).days + 1
            if total_dias_historia == 0: return {}

            for temp in ['🔥 Caliente', '🟡 Tibio', '🧊 Frío']:
                numeros_grupo = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == temp]['Numero'].tolist()
                if not numeros_grupo: continue
                    
                apariciones_historicas_grupo = df_historial_filtrado[df_historial_filtrado['Numero'].isin(numeros_grupo)]
                frecuencia_historica = len(apariciones_historicas_grupo) / total_dias_historia

                fecha_inicio_reciente = fecha_referencia - pd.Timedelta(days=dias_recientes)
                df_reciente = df_historial_filtrado[df_historial_filtrado['Fecha'] >= fecha_inicio_reciente]
                apariciones_recientes_grupo = df_reciente[df_reciente['Numero'].isin(numeros_grupo)]
                frecuencia_reciente = len(apariciones_recientes_grupo) / dias_recientes

                ratio_rendimiento = frecuencia_reciente / frecuencia_historica if frecuencia_historica > 0 else 0
                
                if ratio_rendimiento < 0.7: estado_grupo = "Vencido (Enfriado)"
                elif ratio_rendimiento > 1.3: estado_grupo = "Sobrecalentado"
                else: estado_grupo = "Normal"
                    
                resultados[temp] = {
                    'Estado': estado_grupo, 'Frecuencia Histórica (por día)': round(frecuencia_historica, 2),
                    'Frecuencia Reciente (por día)': round(frecuencia_reciente, 2),
                    'Ratio Rendimiento (%)': round(ratio_rendimiento * 100, 1),
                    'Total Apariciones (7 días)': len(apariciones_recientes_grupo), 'Total Números': len(numeros_grupo)
                }
            return resultados

        grupos_estado_corregido = calcular_estado_grupo_corregido(df_analisis, df_clasificacion_actual, fecha_referencia)
        
        # --- SECCIÓN 3.5: SUGERENCIA ESTRATÉGICA ---
        st.markdown("---")
        st.header("💡 Sugerencia Estratégica Basada en Tendencias (Prioridad por Grupo)")
        
        if grupos_estado_corregido:
            priority_map = {
                "Vencido (Enfriado)": 1,
                "Normal": 2,
                "Sobrecalentado": 3
            }
            
            df_grupos_estado = pd.DataFrame.from_dict(grupos_estado_corregido, orient='index').reset_index()
            df_grupos_estado.rename(columns={'index': 'Grupo'}, inplace=True)
            
            df_grupos_estado['Prioridad'] = df_grupos_estado['Estado'].map(priority_map)
            df_grupos_estado = df_grupos_estado.sort_values(by='Prioridad')
            
            st.subheader("Resumen de Estado de Grupos")
            st.dataframe(df_grupos_estado[['Grupo', 'Estado', 'Prioridad']], width='stretch', hide_index=True)
            
            st.markdown("---")
            st.subheader("Top 5 Candidatos por Grupo (Ordenados por Prioridad)")
            
            for index, row in df_grupos_estado.iterrows():
                grupo = row['Grupo']
                estado = row['Estado']
                
                st.markdown(f"### 🏆 Prioridad {row['Prioridad']}: {grupo} ({estado})")
                
                numeros_grupo = df_clasificacion_actual[df_clasificacion_actual['Temperatura'] == grupo]['Numero'].tolist()
                df_grupo_con_estados = df_estados_completos[df_estados_completos['Numero'].isin(numeros_grupo)]
                
                df_grupo_vencidos = df_grupo_con_estados[df_grupo_con_estados['Estado_Numero'].isin(['Vencido', 'Muy Vencido'])]
                
                top_5_lista = []
                
                if not df_grupo_vencidos.empty:
                    df_grupo_vencidos = df_grupo_vencidos.sort_values(by='Salto_Numero', ascending=False)
                    top_5_numeros = df_grupo_vencidos.head(5)['Numero'].tolist()
                    
                    st.success(f"Oportunidad (Vencidos): {', '.join([f'{n:02d}' for n in top_5_numeros])}")
                    top_5_lista = top_5_numeros
                else:
                    top_5_normales = df_grupo_con_estados.head(5)['Numero'].tolist()
                    st.info(f"Normales (Top 5): {', '.join([f'{n:02d}' for n in top_5_normales])}")
                    top_5_lista = top_5_normales
                
                st.markdown("---")
        else: 
            st.warning("No se pudo calcular el estado de los grupos.")

        # --- SECCIÓN 4: ANÁLISIS DE OPORTUNIDAD POR DÍGITO ---
        st.markdown("---")
        st.header("🎯 Análisis de Oportunidad por Dígito (Decenas y Unidades)")
        col_dec, col_uni = st.columns(2)
        with col_dec:
            st.subheader("📊 Oportunidad por Decena")
            st.dataframe(df_oportunidad_decenas.sort_values(by='Puntuación Total', ascending=False), width='stretch', hide_index=True)
        with col_uni:
            st.subheader("📊 Oportunidad por Unidad")
            st.dataframe(df_oportunidad_unidades.sort_values(by='Puntuación Total', ascending=False), width='stretch', hide_index=True)
        
        # --- SECCIÓN 6: PATRONES SECUENCIALES ---
        st.markdown("---")
        st.header("🔍 Búsqueda de Patrones Secuenciales")
        df_analisis_para_patrones = df_analisis[df_analisis['Fecha'] < fecha_referencia].copy()
        nombre_sesion_para_patrones = modo_sorteo.split(':')[-1].strip()
        patrones = buscar_patrones_secuenciales(df_analisis_para_patrones, max_longitud=3, nombre_sesion=nombre_sesion_para_patrones)
        
        if patrones:
            st.subheader("Últimos Patrones Detectados y Posibles Siguientes")
            df_fijo = df_analisis_para_patrones[df_analisis_para_patrones['Posicion'] == 'Fijo'].copy()
            ultimos_numeros = df_fijo.tail(3)['Numero'].tolist()
            
            if len(ultimos_numeros) >= 2:
                patron_2 = tuple(ultimos_numeros[-2:])
                st.write(f"**Último patrón de 2 números:** {patron_2[0]} → {patron_2[1]}")
                if patron_2 in patrones:
                    siguientes_2 = patrones[patron_2][:3]
                    df_siguientes_2 = pd.DataFrame(siguientes_2, columns=['Siguiente Número', 'Frecuencia'])
                    st.dataframe(df_siguientes_2, width='stretch', hide_index=True)
                    if siguientes_2:
                        recomendacion_2 = siguientes_2[0][0]
                        st.success(f"Próximo probable: **{recomendacion_2:02d}**")
                else:
                    st.warning(f"El patrón reciente `{patron_2[0]} → {patron_2[1]}` no se ha repetido.")
            
            if len(ultimos_numeros) >= 3:
                patron_3 = tuple(ultimos_numeros[-3:])
                st.write(f"**Último patrón de 3 números:** {patron_3[0]} → {patron_3[1]} → {patron_3[2]}")
                if patron_3 in patrones:
                    siguientes_3 = patrones[patron_3][:3]
                    df_siguientes_3 = pd.DataFrame(siguientes_3, columns=['Siguiente Número', 'Frecuencia'])
                    st.dataframe(df_siguientes_3, width='stretch', hide_index=True)
                    if siguientes_3:
                        recomendacion_3 = siguientes_3[0][0]
                        st.success(f"Próximo probable: **{recomendacion_3:02d}**")
                else:
                    st.warning(f"El patrón reciente `{patron_3[0]} → {patron_3[1]} → {patron_3[2]}` no se ha repetido.")
            
            st.markdown("---"); st.subheader("Patrones Más Frecuentes")
            frecuencia_patrones = {pat: sum(f for _, f in sigs) for pat, sigs in patrones.items()}
            patrones_ordenados = sorted(frecuencia_patrones.items(), key=lambda x: x[1], reverse=True)
            top_patrones = patrones_ordenados[:10]
            df_top_patrones = pd.DataFrame(top_patrones, columns=['Patrón', 'Frecuencia Total'])
            df_top_patrones['Patrón'] = df_top_patrones['Patrón'].apply(lambda x: ' → '.join([str(n) for n in x]))
            st.dataframe(df_top_patrones, width='stretch', hide_index=True)
        else: 
            st.warning("No se encontraron patrones.")
        
        # --- SECCIÓN 8: COMPORTAMIENTO DOBLE NORMAL ---
        st.markdown("---")
        st.header("📊 Análisis de Comportamiento del 'Doble Normal'")
        df_comportamiento = analizar_comportamiento_doble_normal(df_analisis)
        if not df_comportamiento.empty:
            st.dataframe(df_comportamiento, width='stretch', hide_index=True)
        
        # --- SECCIÓN 9: MATRIZ COMPLETA Y CANDIDATOS ---
        st.markdown("---")
        st.header("🧠 Generador y Evaluador de Estrategia de Tendencia")
        
        st.subheader(f"Candidatos para el {fecha_referencia.strftime('%d/%m/%Y')}")
        
        df_completa, candidatos_finales, candidatos_vencidos, candidatos_normales = generar_estrategia_tendencia(df_analisis, fecha_referencia)
        
        st.markdown("---")
        st.subheader("🔵 Matriz Completa: Todos los números 'Doble Normal'")
        
        if not df_completa.empty:
            st.dataframe(df_completa, use_container_width=True)
        else:
            st.info("No hay números 'Doble Normal' para hoy.")
        
        st.markdown("---")
        if candidatos_vencidos:
            st.warning(f"🔥 **Candidatos Urgentes (Doble Normal Vencidos)**: {len(candidatos_vencidos)} números.")
            st.write(", ".join([f"{n:02d}" for n in candidatos_vencidos]))
        else:
            st.info("No hay candidatos 'Doble Normal Vencidos' para hoy.")
            
        if candidatos_normales:
            st.success(f"🧊 **Candidatos Estables (Doble Normal Rendimiento Semanal)**: {len(candidatos_normales)} números.")
            st.write(", ".join([f"{n:02d}" for n in candidatos_normales]))
        else:
            st.info("No hay candidatos 'Doble Normal Estables' para hoy.")

        if candidatos_finales:
            st.markdown("---")
            st.info(f"📋 **Lista Unificada ({len(candidatos_finales)} números):**")
            st.write(", ".join([f"{n:02d}" for n in candidatos_finales]))
        else:
            st.warning("La estrategia no encuentra candidatos que cumplan los criterios para hoy.")

        # --- SECCIÓN 10: HISTORIAL VISUAL ENRIQUECIDA ---
        st.markdown("---")
        st.header("📊 Historial Visual Enriquecido (Últimos 30 Fijos)")
        
        df_historico_visual = crear_tabla_historico_visual_fijo(df_analisis)
        if not df_historico_visual.empty:
            total_sorteos = len(df_historico_visual)
            aciertos = df_historico_visual['Es Doble Normal'].sum()
            tasa_acierto = (aciertos / total_sorteos * 100) if total_sorteos > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de sorteos analizados", total_sorteos)
            col2.metric("Aciertos 'Doble Normal'", aciertos)
            col3.metric("Tasa de Acierto", f"{tasa_acierto:.1f}%")
            
            def resaltar_fila_doble_normal(row):
                return ['background-color: #d4edda' if row['Es Doble Normal'] else '' for _ in row]
            
            st.dataframe(df_historico_visual.style.apply(resaltar_fila_doble_normal, axis=1), width='stretch', hide_index=True)

        # --- SECCIÓN 11: COMPARACIÓN DE SESIONES ---
        if modo_sorteo != "Análisis General (Todos los sorteos)":
            st.markdown("---")
            st.header("🔍 Comparación de Sesiones")

            top_10_actual = df_clasificacion_actual.head(10)
            numeros_top_10 = top_10_actual['Numero'].tolist()

            sesion_actual_key = modo_sorteo.split(':')[-1].strip()
            otras_sesiones = [k for k in ['M', 'T', 'N'] if k != sesion_actual_key]
            
            datos_comparacion = []
            for num in numeros_top_10:
                fila = {'Número': f"{num:02d}"}
                freq_actual = df_analisis[df_analisis['Numero'] == num].shape[0]
                fila[f'Frecuencia en {modo_sorteo.split(":")[-1].strip()}'] = freq_actual
                for key in otras_sesiones:
                    df_otra_sesion = df_historial_completo[(df_historial_completo['Tipo_Sorteo'] == key) & (df_historial_completo['Posicion'] == 'Fijo')]
                    freq_otra = df_otra_sesion[df_otra_sesion['Numero'] == num].shape[0]
                    fila[f'Frecuencia en {key}'] = freq_otra
                freq_general = df_historial_completo[df_historial_completo['Posicion'] == 'Fijo']['Numero'].value_counts().get(num, 0)
                fila['Frecuencia General'] = freq_general
                temp_general = df_clasificacion_general[df_clasificacion_general['Numero'] == num]['Temperatura'].iloc[0]
                fila['Temperatura General'] = temp_general
                
                datos_comparacion.append(fila)
                
            df_comparacion = pd.DataFrame(datos_comparacion)
            st.dataframe(df_comparacion, width='stretch', hide_index=True)

if __name__ == "__main__":
    main()