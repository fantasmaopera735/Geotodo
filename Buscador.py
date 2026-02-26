# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import calendar
from collections import Counter
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

RUTA_CSV = 'Geotodo.csv'

st.set_page_config(
    page_title="Geotodo - Suite Ultimate",
    page_icon="🍑", # Logo Melocotón
    layout="wide"
)

st.title("🍑 Geotodo - Suite Ultimate")

@st.cache_resource
def cargar_datos_geotodo(_ruta_csv):
    try:
        if not os.path.exists(_ruta_csv):
            st.error(f"No se encontro el archivo {_ruta_csv}")
            st.stop()
        
        with open(_ruta_csv, 'r', encoding='utf-8-sig') as f:
            primera_linea = f.readline()
        
        if ',' in primera_linea and ';' not in primera_linea:
            separador = ','
        elif ';' in primera_linea:
            separador = ';'
        else:
            separador = ';'
        
        # on_bad_lines='skip' evita errores si hay columnas de más
        df = pd.read_csv(_ruta_csv, sep=separador, encoding='utf-8-sig', dtype=str, on_bad_lines='skip')
        
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
        st.info(f"Columnas detectadas: {list(df.columns)}")
        
        col_fecha = col_tipo = col_centena = col_fijo = col_1er = col_2do = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            col_clean = col_lower.replace('/', '').replace('_', '').replace('-', '')
            
            if 'fecha' in col_lower:
                col_fecha = col
            if col_tipo is None:
                if 'mañana' in col_lower or 'manana' in col_lower or 'tarde' in col_lower or 'noche' in col_lower:
                    col_tipo = col
                elif 'tipo' in col_lower or 'sorteo' in col_lower:
                    col_tipo = col
                elif 'tardenoche' in col_clean:
                    col_tipo = col
            if col_centena is None and ('centena' in col_lower or 'cent' in col_lower):
                col_centena = col
            if col_fijo is None and col_lower == 'fijo':
                col_fijo = col
            if col_1er is None:
                if '1er' in col_lower or 'primer' in col_lower or '1ercorreo' in col_clean:
                    col_1er = col
            if col_2do is None:
                if '2do' in col_lower or 'segundo' in col_lower or '2docorrido' in col_clean:
                    col_2do = col
        
        num_cols = len(df.columns)
        if col_fecha is None and num_cols >= 1:
            col_fecha = df.columns[0]
        if col_tipo is None and num_cols >= 2:
            col_tipo = df.columns[1]
        
        if num_cols >= 6:
            if col_centena is None: col_centena = df.columns[2]
            if col_fijo is None: col_fijo = df.columns[3]
            if col_1er is None: col_1er = df.columns[4]
            if col_2do is None: col_2do = df.columns[5]
        elif num_cols >= 5:
            if col_fijo is None: col_fijo = df.columns[2]
            if col_1er is None: col_1er = df.columns[3]
            if col_2do is None: col_2do = df.columns[4]
        
        mapeo = {}
        if col_fecha: mapeo[col_fecha] = 'Fecha'
        if col_tipo: mapeo[col_tipo] = 'Tipo_Sorteo'
        if col_centena: mapeo[col_centena] = 'Centena'
        if col_fijo: mapeo[col_fijo] = 'Fijo'
        if col_1er: mapeo[col_1er] = 'Primer_Corrido'
        if col_2do: mapeo[col_2do] = 'Segundo_Corrido'
        
        df = df.rename(columns=mapeo)
        if 'Centena' not in df.columns:
            df['Centena'] = '0'
        
        def convertir_fecha(valor):
            if pd.isna(valor) or str(valor).strip() == '':
                return pd.NaT
            valor_str = str(valor).strip()
            formatos = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y', '%Y/%m/%d']
            for fmt in formatos:
                try:
                    return pd.to_datetime(valor_str, format=fmt)
                except:
                    continue
            try:
                return pd.to_datetime(valor_str, dayfirst=True)
            except:
                return pd.NaT
        
        df['Fecha'] = df['Fecha'].apply(convertir_fecha)
        fechas_invalidas = df['Fecha'].isna().sum()
        if fechas_invalidas > 0:
            st.warning(f"Se encontraron {fechas_invalidas} fechas con formato invalido.")
        df = df.dropna(subset=['Fecha'])
        
        if 'Tipo_Sorteo' in df.columns:
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper()
            # Lógica para 3 sesiones: M, T, N
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].apply(lambda x: 
                'M' if x in ['M', 'MAÑANA', 'MANANA', 'MAÑANA/', 'M/'] else
                'T' if x in ['T', 'TARDE', 'TARDE/'] else
                'N' if x in ['N', 'NOCHE', '/NOCHE', 'NOCHE/'] else
                'M' if 'MAÑANA' in x or 'MANANA' in x else
                'T' if 'TARDE' in x else 'N' if 'NOCHE' in x else 'OTRO')
        else:
            df['Tipo_Sorteo'] = 'OTRO'
        
        if 'Fijo' in df.columns:
            df_fijos = df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy()
            df_fijos = df_fijos.rename(columns={'Fijo': 'Numero'})
            df_fijos['Numero'] = pd.to_numeric(df_fijos['Numero'], errors='coerce')
            df_fijos = df_fijos.dropna(subset=['Numero'])
            df_fijos['Numero'] = df_fijos['Numero'].astype(int)
        else:
            st.error("No se encontro la columna Fijo")
            st.stop()
        
        # Ordenamiento: Mañana (0) -> Tarde (1) -> Noche (2)
        draw_order_map = {'M': 0, 'T': 1, 'N': 2, 'OTRO': 3}
        df_fijos['draw_order'] = df_fijos['Tipo_Sorteo'].map(draw_order_map).fillna(3)
        df_fijos['sort_key'] = df_fijos['Fecha'] + pd.to_timedelta(df_fijos['draw_order'], unit='h')
        df_fijos = df_fijos.sort_values(by='sort_key').reset_index(drop=True)
        
        return df_fijos, df
        
    except pd.errors.EmptyDataError:
        st.error("El archivo CSV esta vacio.")
        st.stop()
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

def extraer_digitos_sesion(centena, fijo, primer_corr, segundo_corr):
    digitos = {
        'centena': [], 'fijo_dec': [], 'fijo_uni': [], 'corrido1_dec': [], 
        'corrido1_uni': [], 'corrido2_dec': [], 'corrido2_uni': [], 'todos': []
    }
    try:
        c = int(float(centena))
        digitos['centena'].append(c)
        digitos['todos'].append(c)
    except: pass
    try:
        f = int(float(fijo))
        digitos['fijo_dec'].append(f // 10)
        digitos['fijo_uni'].append(f % 10)
        digitos['todos'].extend([f // 10, f % 10])
    except: pass
    try:
        p1 = int(float(primer_corr))
        digitos['corrido1_dec'].append(p1 // 10)
        digitos['corrido1_uni'].append(p1 % 10)
        digitos['todos'].extend([p1 // 10, p1 % 10])
    except: pass
    try:
        p2 = int(float(segundo_corr))
        digitos['corrido2_dec'].append(p2 // 10)
        digitos['corrido2_uni'].append(p2 % 10)
        digitos['todos'].extend([p2 // 10, p2 % 10])
    except: pass
    return digitos

def analizar_dia_completo(df_completo, fecha):
    df_dia = df_completo[df_completo['Fecha'].dt.date == fecha.date()].copy()
    if df_dia.empty:
        return None, "Sin datos para esa fecha"
    
    todos_digitos = []
    digitos_centena = []
    digitos_fijo = []
    digitos_corr1 = []
    digitos_corr2 = []
    detalle_digitos = []
    
    for _, row in df_dia.iterrows():
        centena = row.get('Centena', 0)
        fijo = row.get('Fijo', 0)
        primer_corr = row.get('Primer_Corrido', 0)
        segundo_corr = row.get('Segundo_Corrido', 0)
        
        digitos = extraer_digitos_sesion(centena, fijo, primer_corr, segundo_corr)
        todos_digitos.extend(digitos['todos'])
        digitos_centena.extend(digitos['centena'])
        digitos_fijo.extend(digitos['fijo_dec'] + digitos['fijo_uni'])
        digitos_corr1.extend(digitos['corrido1_dec'] + digitos['corrido1_uni'])
        digitos_corr2.extend(digitos['corrido2_dec'] + digitos['corrido2_uni'])
        
        detalle_digitos.append({
            'Sesion': row['Tipo_Sorteo'], 'Centena': centena, 'Fijo': fijo,
            '1er_Corrido': primer_corr, '2do_Corrido': segundo_corr, 'Digitos': digitos['todos']
        })
    
    todos_dig = set(range(10))
    presentes = set(todos_digitos)
    faltantes = todos_dig - presentes
    
    return {
        'digitos_presentes': sorted(list(presentes)),
        'digitos_faltantes': sorted(list(faltantes)),
        'sesiones': [row['Tipo_Sorteo'] for _, row in df_dia.iterrows()],
        'total_digitos': len(todos_digitos),
        'detalle': detalle_digitos,
        'digitos_lista': todos_digitos,
        'por_tipo': {'centena': digitos_centena, 'fijo': digitos_fijo, 
                     'corrido1': digitos_corr1, 'corrido2': digitos_corr2}
    }, None

def estadisticas_digitos_separadas(df_completo, dias_atras=180):
    fecha_hoy = datetime.now()
    fecha_inicio = fecha_hoy - timedelta(days=dias_atras)
    df_filtrado = df_completo[df_completo['Fecha'] >= fecha_inicio].copy()
    
    contadores = {'general': Counter(), 'centena': Counter(), 'fijo': Counter(), 
                  'corrido1': Counter(), 'corrido2': Counter()}
    ultima_aparicion = {'general': {d: None for d in range(10)}, 'centena': {d: None for d in range(10)},
                        'fijo': {d: None for d in range(10)}, 'corrido1': {d: None for d in range(10)},
                        'corrido2': {d: None for d in range(10)}}
    fechas_aparicion = {'general': {d: [] for d in range(10)}, 'centena': {d: [] for d in range(10)},
                        'fijo': {d: [] for d in range(10)}, 'corrido1': {d: [] for d in range(10)},
                        'corrido2': {d: [] for d in range(10)}}
    
    fechas_unicas = sorted(df_filtrado['Fecha'].dt.date.unique())
    
    for fecha in fechas_unicas:
        fecha_dt = datetime.combine(fecha, datetime.min.time())
        resultado, _ = analizar_dia_completo(df_filtrado, fecha_dt)
        
        if resultado:
            for d in resultado['digitos_presentes']:
                contadores['general'][d] += 1
                ultima_aparicion['general'][d] = fecha
                fechas_aparicion['general'][d].append(fecha)
            for d in resultado['por_tipo']['centena']:
                contadores['centena'][d] += 1
                ultima_aparicion['centena'][d] = fecha
                fechas_aparicion['centena'][d].append(fecha)
            for d in resultado['por_tipo']['fijo']:
                contadores['fijo'][d] += 1
                ultima_aparicion['fijo'][d] = fecha
                fechas_aparicion['fijo'][d].append(fecha)
            for d in resultado['por_tipo']['corrido1']:
                contadores['corrido1'][d] += 1
                ultima_aparicion['corrido1'][d] = fecha
                fechas_aparicion['corrido1'][d].append(fecha)
            for d in resultado['por_tipo']['corrido2']:
                contadores['corrido2'][d] += 1
                ultima_aparicion['corrido2'][d] = fecha
                fechas_aparicion['corrido2'][d].append(fecha)
    
    fecha_hoy_date = fecha_hoy.date()
    stats = {}
    
    for tipo in ['general', 'centena', 'fijo', 'corrido1', 'corrido2']:
        datos = []
        total_sorteos = sum(contadores[tipo].values()) if contadores[tipo] else 1
        
        for d in range(10):
            freq = contadores[tipo].get(d, 0)
            ultima = ultima_aparicion[tipo][d]
            dias_sin = (fecha_hoy_date - ultima).days if ultima else 999
            porcentaje = round((freq / total_sorteos) * 100, 1) if total_sorteos > 0 else 0
            
            fechas_d = fechas_aparicion[tipo][d]
            if len(fechas_d) >= 2:
                gaps = [(fechas_d[i+1] - fechas_d[i]).days for i in range(len(fechas_d)-1)]
                promedio_salida = round(np.mean(gaps), 1)
                ausencia_maxima = max(gaps)
            elif len(fechas_d) == 1:
                promedio_salida = dias_sin
                ausencia_maxima = dias_sin
            else:
                promedio_salida = 999
                ausencia_maxima = 999
            
            datos.append({
                'Digito': d, 'Frecuencia': freq, 'Porcentaje': f"{porcentaje}%",
                'Promedio_Salida': promedio_salida if fechas_d else 'N/A',
                'Ausencia_Maxima': ausencia_maxima if fechas_d else 'N/A',
                'Dias_Sin_Aparecer': dias_sin if ultima else 'N/A',
                'Ultima': ultima.strftime('%d/%m') if ultima else 'N/A'
            })
        stats[tipo] = pd.DataFrame(datos)
    return stats

def backtest_digito_faltante(df_completo, dias_atras=90):
    fecha_hoy = datetime.now()
    fecha_inicio = fecha_hoy - timedelta(days=dias_atras)
    fechas_unicas = sorted(df_completo['Fecha'].dt.date.unique())
    
    resultados = []
    aciertos = 0
    total_evaluados = 0
    
    for i, fecha in enumerate(fechas_unicas):
        fecha_dt = datetime.combine(fecha, datetime.min.time())
        if i >= len(fechas_unicas) - 1: continue
        fecha_siguiente = fechas_unicas[i + 1]
        if fecha_dt < fecha_inicio: continue
        
        resultado_dia, error = analizar_dia_completo(df_completo, fecha_dt)
        if error or not resultado_dia['digitos_faltantes']: continue
        
        faltantes = resultado_dia['digitos_faltantes']
        df_siguiente = df_completo[df_completo['Fecha'].dt.date == fecha_siguiente]
        if df_siguiente.empty: continue
        
        fijos_siguiente = df_siguiente['Fijo'].tolist()
        digitos_fijos_siguiente = set()
        for f in fijos_siguiente:
            try:
                f_int = int(float(f))
                digitos_fijos_siguiente.add(f_int // 10)
                digitos_fijos_siguiente.add(f_int % 10)
            except: pass
        
        coincidencias = [d for d in faltantes if d in digitos_fijos_siguiente]
        acierto = len(coincidencias) > 0
        if acierto: aciertos += 1
        total_evaluados += 1
        
        resultados.append({
            'Fecha': fecha, 'Faltantes': ','.join(map(str, faltantes)),
            'Fijos_Sig': ','.join([f"{int(float(f)):02d}" for f in fijos_siguiente]),
            'Coincidencia': 'SI' if acierto else 'NO',
            'Digitos_Coinc': ','.join(map(str, coincidencias)) if coincidencias else '-'
        })
    
    efectividad = (aciertos / total_evaluados * 100) if total_evaluados > 0 else 0
    return {'resultados': resultados, 'total_evaluados': total_evaluados, 'aciertos': aciertos, 'efectividad': round(efectividad, 2)}

def analizar_siguientes(df_fijos, numero_busqueda, ventana_sorteos):
    indices = df_fijos[df_fijos['Numero'] == numero_busqueda].index.tolist()
    if not indices: return None, 0
    lista_s = []
    for idx in indices:
        i, f = idx + 1, idx + ventana_sorteos + 1
        if i < len(df_fijos): lista_s.extend(df_fijos.iloc[i:f]['Numero'].tolist())
    if not lista_s: return None, len(indices)
    c = Counter(lista_s)
    r = pd.DataFrame.from_dict(c, orient='index', columns=['Frecuencia'])
    r['Probabilidad'] = (r['Frecuencia'] / len(lista_s) * 100).round(2)
    r['Numero'] = [f"{int(x):02d}" for x in r.index]
    return r.sort_values('Frecuencia', ascending=False), len(indices)

def analizar_almanaque(df_fijos, dia_inicio, dia_fin, meses_atras, strict_mode=True):
    fecha_hoy = datetime.now()
    bloques_validos = []
    nombres_bloques = []
    
    for offset in range(1, meses_atras + 1):
        f_obj = fecha_hoy - relativedelta(months=offset)
        try:
            last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
            f_i = datetime(f_obj.year, f_obj.month, min(dia_inicio, last_day))
            f_f = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
            if f_i > f_f: continue
            df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
            if not df_b.empty:
                bloques_validos.append(df_b)
                nombres_bloques.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
        except: continue
    
    if not bloques_validos:
        for offset in range(1, meses_atras + 1):
            f_obj = fecha_hoy - relativedelta(months=offset)
            try:
                f_i = datetime(f_obj.year, f_obj.month, 1)
                last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
                f_f = datetime(f_obj.year, f_obj.month, last_day)
                df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
                if not df_b.empty:
                    bloques_validos.append(df_b)
                    nombres_bloques.append(f"{f_obj.strftime('%b')} (Todo el mes)")
            except: continue
        if not bloques_validos: return {'success': False, 'mensaje': 'Sin datos'}
    
    df_total = pd.concat(bloques_validos)
    df_total['Decena'] = df_total['Numero'] // 10
    df_total['Unidad'] = df_total['Numero'] % 10
    
    cnt_d = df_total['Decena'].value_counts().reindex(range(10), fill_value=0)
    cnt_u = df_total['Unidad'].value_counts().reindex(range(10), fill_value=0)
    
    def clasificar(serie):
        df_t = serie.sort_values(ascending=False).reset_index()
        df_t.columns = ['Digito', 'Frecuencia']
        conds = [(df_t.index < 3), (df_t.index < 6)]
        vals = ['🔥 Caliente', '🟡 Tibio']
        df_t['Estado'] = np.select(conds, vals, default='🧊 Frío')
        mapa = {r['Digito']: r['Estado'] for _, r in df_t.iterrows()}
        return df_t, mapa

    df_dec, mapa_d = clasificar(cnt_d)
    df_uni, mapa_u = clasificar(cnt_u)
    
    hot_d = df_dec[df_dec['Estado'] == '🔥 Caliente']['Digito'].tolist()
    hot_u = df_uni[df_uni['Estado'] == '🔥 Caliente']['Digito'].tolist()
    lista_3x3 = [{'Número': f"{d*10+u:02d}", 'Veces': len(df_total[df_total['Numero'] == d*10+u])} 
                 for d in hot_d for u in hot_u]
    df_3x3 = pd.DataFrame(lista_3x3).sort_values('Veces', ascending=False) if lista_3x3 else pd.DataFrame(columns=['Número', 'Veces'])

    ranking = []
    for n, v in df_total['Numero'].value_counts().items():
        d, u = n // 10, n % 10
        ranking.append({'Número': f"{n:02d}", 'Frecuencia': v, 
                       'Perfil': f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}"})
    df_rank = pd.DataFrame(ranking).sort_values('Frecuencia', ascending=False) if ranking else pd.DataFrame(columns=['Número', 'Frecuencia', 'Perfil'])
    
    tend = df_rank['Perfil'].value_counts().reset_index()
    tend.columns = ['Perfil', 'Frecuencia']
    top_p = tend.iloc[0]['Perfil'] if not tend.empty else "N/A"

    tend_nums = []
    if top_p != "N/A" and " + " in top_p:
        p_dec, p_uni = top_p.split(" + ")
        decs_obj = df_dec[df_dec['Estado'] == p_dec]['Digito'].tolist()
        unis_obj = df_uni[df_uni['Estado'] == p_uni]['Digito'].tolist()
        for d in decs_obj:
            for u in unis_obj:
                tend_nums.append({'Número': f"{d*10+u:02d}", 'Sugerencia': f"{p_dec} x {p_uni}"})
    df_tend_nums = pd.DataFrame(tend_nums)

    pers_num = []
    nums_unicos = df_total['Numero'].unique()
    for n in nums_unicos:
        c = sum(1 for b in bloques_validos if n in b['Numero'].values)
        if c == len(bloques_validos):
            perfil_val = df_rank[df_rank['Número'] == f"{n:02d}"]['Perfil']
            p = perfil_val.values[0] if not perfil_val.empty else "Desconocido"
            pers_num.append({'Número': f"{n:02d}", 'Perfil': p})
    df_pers_num = pd.DataFrame(pers_num).sort_values('Número').reset_index(drop=True) if pers_num else pd.DataFrame(columns=['Número', 'Perfil'])

    sets_perfiles = []
    for df_b in bloques_validos:
        perfiles_en_bloque = set()
        for row in df_b.itertuples():
            d, u = row.Numero // 10, row.Numero % 10
            perfiles_en_bloque.add(f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}")
        sets_perfiles.append(perfiles_en_bloque)
    
    persistentes_perfiles = set.intersection(*sets_perfiles) if sets_perfiles else set()
    persistentes_num_set = set(p['Número'] for p in pers_num) if pers_num else set()

    hoy = datetime.now()
    estado_periodo = ""
    df_historial_actual = pd.DataFrame()
    
    try:
        fin_mes_actual = calendar.monthrange(hoy.year, hoy.month)[1]
        fecha_ini_evaluacion = datetime(hoy.year, hoy.month, min(dia_inicio, fin_mes_actual))
        fecha_fin_teorica = datetime(hoy.year, hoy.month, min(dia_fin, fin_mes_actual))
        
        if hoy < fecha_ini_evaluacion:
            estado_periodo = f"⚪ PERIODO NO INICIADO (Comienza el {fecha_ini_evaluacion.strftime('%d/%m')})"
        else:
            fecha_fin_real = min(hoy, fecha_fin_teorica)
            df_evaluacion = df_fijos[(df_fijos['Fecha'] >= fecha_ini_evaluacion) & (df_fijos['Fecha'] <= fecha_fin_real)].copy()
            
            if not df_evaluacion.empty:
                historial_data = []
                for row in df_evaluacion.itertuples():
                    num = row.Numero
                    d, u = num // 10, num % 10
                    perfil_completo = f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}"
                    cumple_regla = f"{num:02d}" in persistentes_num_set or perfil_completo in persistentes_perfiles
                    motivo = "Num. Persistente" if f"{num:02d}" in persistentes_num_set else ("Perfil Persistente" if perfil_completo in persistentes_perfiles else "")
                    historial_data.append({
                        'Fecha': row.Fecha, 'Tipo_Sorteo': row.Tipo_Sorteo, 'Número': f"{num:02d}",
                        'Perfil (D/U)': perfil_completo,
                        'Cumple Regla': '✅ SÍ' if cumple_regla else '❌ NO',
                        'Tipo Regla': motivo if cumple_regla else '-'
                    })
                df_historial_actual = pd.DataFrame(historial_data)
                # Ordenamiento Geotodo: N(0), T(1), M(2)
                orden_sorteo = {'N': 0, 'T': 1, 'M': 2, 'OTRO': 3}
                df_historial_actual['orden'] = df_historial_actual['Tipo_Sorteo'].map(orden_sorteo).fillna(3)
                df_historial_actual = df_historial_actual.sort_values(['Fecha', 'orden'], ascending=[False, True]).reset_index(drop=True)
                df_historial_actual = df_historial_actual.drop(columns=['orden'])
            estado_periodo = f"🟢 PERIODO ACTIVO (Evaluado hasta: {hoy.strftime('%d/%m')})"
    except Exception as e:
        estado_periodo = f"⚪ Error: {str(e)}"

    df_faltantes = pd.DataFrame()
    if "ACTIVO" in estado_periodo:
        esperados = set(df_rank.head(20)['Número'].tolist()) if not df_rank.empty else set()
        esperados.update(persistentes_num_set)
        if not df_historial_actual.empty:
            salidos = set(df_historial_actual['Número'].unique())
            faltantes_nums = esperados - salidos
        else:
            faltantes_nums = esperados
        if faltantes_nums:
            df_faltantes = pd.DataFrame([{'Número': n, 'Estado': '⏳ FALTANTE'} for n in sorted(list(faltantes_nums))])

    return {
        'success': True, 'df_total': df_total, 'df_dec': df_dec, 'df_uni': df_uni,
        'df_3x3': df_3x3, 'df_rank': df_rank, 'nombres_bloques': nombres_bloques,
        'df_pers_num': df_pers_num, 'tend': tend, 'top_p': top_p, 'df_tend_nums': df_tend_nums,
        'persistentes_perfiles': persistentes_perfiles, 'df_historial_actual': df_historial_actual,
        'df_faltantes': df_faltantes, 'estado_periodo': estado_periodo
    }

def backtesting_estrategia(df_fijos, mes_objetivo, anio_objetivo, dia_ini, dia_fin, meses_atras):
    try:
        fecha_ref = datetime(anio_objetivo, mes_objetivo, 1)
        bloques_train = []
        nombres_train = []
        
        for offset in range(1, meses_atras + 1):
            f_obj = fecha_ref - relativedelta(months=offset)
            try:
                last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
                f_i = datetime(f_obj.year, f_obj.month, min(dia_ini, last_day))
                f_f = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
                if f_i > f_f: continue
                df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
                if not df_b.empty:
                    bloques_train.append(df_b)
                    nombres_train.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
            except: continue
        
        if not bloques_train: return None, "Sin datos para entrenamiento"
        
        df_train = pd.concat(bloques_train)
        df_train['Dec'] = df_train['Numero'] // 10
        df_train['Uni'] = df_train['Numero'] % 10
        cnt_d = df_train['Dec'].value_counts().reindex(range(10), fill_value=0)
        cnt_u = df_train['Uni'].value_counts().reindex(range(10), fill_value=0)
        
        def get_lists(serie):
            df_t = serie.sort_values(ascending=False).reset_index()
            df_t.columns = ['Digito', 'Frecuencia']
            conds = [(df_t.index < 3), (df_t.index < 6)]
            vals = ['🔥 Caliente', '🟡 Tibio']
            df_t['Estado'] = np.select(conds, vals, default='🧊 Frío')
            mapa = {r['Digito']: r['Estado'] for _, r in df_t.iterrows()}
            hot = [r['Digito'] for _, r in df_t.iterrows() if 'Caliente' in r['Estado']]
            warm = [r['Digito'] for _, r in df_t.iterrows() if 'Tibio' in r['Estado']]
            cold = [r['Digito'] for _, r in df_t.iterrows() if 'Frío' in r['Estado']]
            return mapa, hot, warm, cold
        
        mapa_d, hot_d, warm_d, cold_d = get_lists(cnt_d)
        mapa_u, hot_u, warm_u, cold_u = get_lists(cnt_u)
        
        sets_perfiles = []
        for df_b in bloques_train:
            perfiles = set()
            for row in df_b.itertuples():
                d, u = row.Numero // 10, row.Numero % 10
                perfiles.add(f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}")
            sets_perfiles.append(perfiles)
        perfiles_persistentes = set.intersection(*sets_perfiles) if sets_perfiles else set()
        
        f_prueba_ini = datetime(anio_objetivo, mes_objetivo, min(dia_ini, 28))
        f_prueba_fin = datetime(anio_objetivo, mes_objetivo, min(dia_fin, calendar.monthrange(anio_objetivo, mes_objetivo)[1]))
        df_test = df_fijos[(df_fijos['Fecha'] >= f_prueba_ini) & (df_fijos['Fecha'] <= f_prueba_fin)]
        
        if df_test.empty: return None, "Sin datos en el mes de prueba"
        
        resultados = []
        aciertos = 0
        sufrientes = 0
        
        for row in df_test.itertuples():
            num = row.Numero
            d, u = num // 10, num % 10
            perfil = f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}"
            es_pers = perfil in perfiles_persistentes
            if es_pers: aciertos += 1
            es_sufriente = (d in cold_d) or (u in cold_u)
            if es_sufriente: sufrientes += 1
            resultados.append({
                'Fecha': row.Fecha, 'Tipo': row.Tipo_Sorteo, 'Numero': num,
                'Decena': f"{d} ({mapa_d.get(d, '?')})",
                'Unidad': f"{u} ({mapa_u.get(u, '?')})",
                'Resultado': 'ESTRUCTURA' if es_pers else ('SUFRIENTE' if es_sufriente else 'OTRO')
            })
        
        df_detalle = pd.DataFrame(resultados)
        orden_sorteo = {'N': 0, 'T': 1, 'M': 2, 'OTRO': 3}
        df_detalle['orden'] = df_detalle['Tipo'].map(orden_sorteo).fillna(3)
        df_detalle = df_detalle.sort_values(['Fecha', 'orden'], ascending=[False, True]).reset_index(drop=True)
        
        total = len(df_detalle)
        porc = (aciertos / total * 100) if total > 0 else 0
        
        return {
            "Entrenamiento": ", ".join(nombres_train), "Prueba": f"{f_prueba_ini.strftime('%B %Y')} ({dia_ini}-{dia_fin})",
            "Perfiles": perfiles_persistentes, "Total": total, "Aciertos": aciertos,
            "Efectividad": round(porc, 2), "Sufrientes": sufrientes, "Detalle": df_detalle,
            "hot_d": hot_d, "warm_d": warm_d, "cold_d": cold_d,
            "hot_u": hot_u, "warm_u": warm_u, "cold_u": cold_u
        }
    except Exception as e:
        return None, f"Error: {str(e)}"

def analizar_estabilidad_numeros(df_fijos, dias_analisis=365):
    fecha_limite = datetime.now() - timedelta(days=dias_analisis)
    df_historico = df_fijos[df_fijos['Fecha'] >= fecha_limite].copy()
    if df_historico.empty: return None
    
    estabilidad_data = []
    hoy = datetime.now()
    
    for num in range(100):
        df_num = df_historico[df_historico['Numero'] == num].sort_values('Fecha')
        
        if len(df_num) < 2:
            max_gap = 9999
            avg_gap = 9999
            std_gap = 0
            gap_actual = (hoy - df_num['Fecha'].max()).days if not df_num.empty else dias_analisis
            estado = "SIN DATOS"
            ultima_fecha = df_num['Fecha'].max() if not df_num.empty else None
        else:
            fechas = df_num['Fecha'].tolist()
            gaps = [(fechas[i+1] - fechas[i]).days for i in range(len(fechas)-1)]
            max_gap = max(gaps) if gaps else 9999
            avg_gap = np.mean(gaps) if gaps else 0
            std_gap = np.std(gaps) if gaps else 0
            ultima_salida = fechas[-1]
            gap_actual = (hoy - ultima_salida).days
            if gap_actual > max_gap: max_gap = gap_actual
            
            if gap_actual == 0:
                estado = "🔥 EN RACHA"
            elif gap_actual <= avg_gap:
                estado = "✅ NORMAL"
            elif gap_actual <= avg_gap * 2.0:
                estado = "⏳ VENCIDO"
            else:
                estado = "🔴 MUY VENCIDO"
            ultima_fecha = ultima_salida

        estabilidad_data.append({
            'Número': f"{num:02d}", 'Gap Actual': gap_actual,
            'Gap Máximo (Días)': max_gap, 'Gap Promedio': round(avg_gap, 1),
            'Desviación (Irregularidad)': round(std_gap, 1), 'Estado': estado,
            'Última Salida': ultima_fecha.strftime('%d/%m/%Y') if ultima_fecha else "N/A"
        })
    
    df_est = pd.DataFrame(estabilidad_data)
    df_est = df_est.sort_values(by=['Gap Máximo (Días)', 'Desviación (Irregularidad)'], ascending=[True, True]).reset_index(drop=True)
    return df_est

def analizar_faltantes_mes(df_fijos, mes, anio, sorteos_freq):
    hoy = datetime.now()
    
    fecha_inicio_mes = datetime(anio, mes, 1)
    last_day = calendar.monthrange(anio, mes)[1]
    fecha_fin_mes = datetime(anio, mes, last_day)
    
    if mes == hoy.month and anio == hoy.year:
        fecha_fin_mes = hoy
    
    df_mes = df_fijos[(df_fijos['Fecha'] >= fecha_inicio_mes) & (df_fijos['Fecha'] <= fecha_fin_mes)]
    salidos = set(df_mes['Numero'].unique())
    faltantes = sorted(list(set(range(100)) - salidos))
    
    if not faltantes:
        return pd.DataFrame(), "Todos los números salieron.", pd.DataFrame()
    
    df_estabilidad = analizar_estabilidad_numeros(df_fijos, 365)
    est_map = {}
    if df_estabilidad is not None:
        for _, row in df_estabilidad.iterrows():
            est_map[row['Número']] = {'Gap': row['Gap Actual'], 'Estado': row['Estado']}

    df_reciente = df_fijos.tail(sorteos_freq)
    conteo = df_reciente['Numero'].value_counts()
    top_frecuencia = conteo.head(25).index.tolist()
    
    resultados = []
    for num in faltantes:
        est_data = est_map.get(f"{num:02d}", {'Gap': 999, 'Estado': 'SIN DATOS'})
        es_vencido = ("VENCIDO" in est_data['Estado'])
        
        es_favorito = (num in top_frecuencia)
        freq_val = conteo.get(num, 0)
        
        if es_vencido or es_favorito:
            prioridad = "🔴 ALTA"
            razones = []
            if es_vencido: razones.append("Atrasado")
            if es_favorito: razones.append("Favorito")
            razon = " + ".join(razones)
        else:
            prioridad = "⚪ BAJA"
            razon = "Sin condiciones"
        
        resultados.append({
            'Número': f"{num:02d}",
            'Prioridad': prioridad,
            'Razón': razon,
            'Veces Salidas': freq_val,
            'Estado Estabilidad': est_data['Estado'],
            'Gap Actual': est_data['Gap']
        })
    
    df_res = pd.DataFrame(resultados)
    ord_map = {"🔴 ALTA": 0, "⚪ BAJA": 1}
    df_res['ord'] = df_res['Prioridad'].map(ord_map)
    df_res = df_res.sort_values(['ord', 'Veces Salidas'], ascending=[True, False]).drop('ord', axis=1)
    
    return df_res, None, df_mes

def generar_sugerencia(df, dias, gap):
    fh = datetime.now()
    df_t = df[df['Fecha'] >= fh - timedelta(days=dias)].copy()
    if df_t.empty: return pd.DataFrame()
    df_t['Dec'] = df_t['Numero'] // 10
    df_t['Uni'] = df_t['Numero'] % 10
    td = df_t['Dec'].value_counts().head(3).index.tolist()
    tu = df_t['Uni'].value_counts().head(3).index.tolist()
    res = []
    for n in [d*10+u for d in td for u in tu]:
        df_n = df[df['Numero'] == n]
        if not df_n.empty:
            g = (fh - df_n['Fecha'].max()).days
            if g >= gap:
                res.append({'Numero': f"{n:02d}", 'Gap': g, 'Estado': "Muy" if g > gap*1.5 else "Op"})
    return pd.DataFrame(res).sort_values('Gap', ascending=False) if res else pd.DataFrame()

def buscar_seq(df_fijos, part, type_, seq):
    try:
        p = [x.strip().upper() for x in seq.replace(',', ' ').split() if x.strip()]
    except: return None, "Error"
    if len(p) == 0 or len(p) > 5: return None, "Invalido"
    if type_ == 'digito': v = set(range(10))
    elif type_ == 'paridad': v = {'P', 'I'}
    elif type_ == 'altura': v = {'A', 'B'}
    else: return None, "Tipo desconocido"
    try:
        if type_ == 'digito':
            p = [int(x) for x in p]
            if any(x not in v for x in p): return None, "0-9"
        else:
            if any(x not in v for x in p): return None, f"Usa: {', '.join(v)}"
    except: return None, "Error conversion"
    l = []
    for x in df_fijos['Numero']:
        val = x // 10 if part == 'Decena' else x % 10
        if type_ == 'digito': l.append(val)
        elif type_ == 'paridad': l.append('P' if val % 2 == 0 else 'I')
        elif type_ == 'altura': l.append('A' if val >= 5 else 'B')
    lp = len(p)
    dat = {}
    for i in range(len(l) - lp):
        if l[i:i+lp] == p:
            sig = l[i + lp]
            r = df_fijos.iloc[i + lp]
            e = f"{r['Numero']:02d} ({r['Fecha'].strftime('%d/%m/%Y')})"
            if sig not in dat: dat[sig] = {'c': 0, 'e': []}
            dat[sig]['c'] += 1
            if len(dat[sig]['e']) < 3 and e not in dat[sig]['e']: dat[sig]['e'].append(e)
    if not dat: return None, "No encontrado"
    total = sum(v['c'] for v in dat.values())
    rows = []
    for k, v in dat.items():
        prob = (v['c'] / total * 100) if total > 0 else 0
        rows.append({'Siguiente': k, 'Frecuencia': v['c'], 'Ejemplos': ", ".join(v['e']), 'Prob': round(prob, 2)})
    return pd.DataFrame(rows).sort_values('Frecuencia', ascending=False), None

def analizar_transferencia_geotodo(df_completo, dias_atras=180):
    fecha_hoy = datetime.now()
    fecha_inicio = fecha_hoy - timedelta(days=dias_atras)
    df_filtrado = df_completo[df_completo['Fecha'] >= fecha_inicio].copy()
    fechas_unicas = sorted(df_filtrado['Fecha'].dt.date.unique())
    
    # Diccionario corregido con las 3 transferencias lógicas
    eventos = {'M->T': [], 'T->N': [], 'N->M': []}
    
    for i, fecha in enumerate(fechas_unicas):
        df_dia = df_filtrado[df_filtrado['Fecha'].dt.date == fecha]
        
        # Extraer valores M, T, N actuales
        fijo_M_vals = df_dia[df_dia['Tipo_Sorteo'] == 'M']['Fijo'].values
        fijo_T_vals = df_dia[df_dia['Tipo_Sorteo'] == 'T']['Fijo'].values
        fijo_N_vals = df_dia[df_dia['Tipo_Sorteo'] == 'N']['Fijo'].values
        
        fijo_M = int(float(fijo_M_vals[0])) if len(fijo_M_vals) > 0 else None
        fijo_T = int(float(fijo_T_vals[0])) if len(fijo_T_vals) > 0 else None
        fijo_N = int(float(fijo_N_vals[0])) if len(fijo_N_vals) > 0 else None
        
        # 1. Lógica M -> T (Mañana a Tarde mismo día)
        if fijo_M is not None and fijo_T is not None:
            decena_M = fijo_M // 10
            unidad_T = fijo_T % 10
            if decena_M == unidad_T:
                eventos['M->T'].append({'fecha': fecha, 'digito': decena_M})
        
        # 2. Lógica T -> N (Tarde a Noche mismo día)
        if fijo_T is not None and fijo_N is not None:
            decena_T = fijo_T // 10
            unidad_N = fijo_N % 10
            if decena_T == unidad_N:
                eventos['T->N'].append({'fecha': fecha, 'digito': decena_T})
        
        # 3. Lógica N -> M (Noche a Mañana día siguiente)
        if fijo_N is not None and i < len(fechas_unicas) - 1:
            fecha_siguiente = fechas_unicas[i + 1]
            df_siguiente = df_filtrado[df_filtrado['Fecha'].dt.date == fecha_siguiente]
            # Buscamos la Mañana del día siguiente
            fijo_M_sig_vals = df_siguiente[df_siguiente['Tipo_Sorteo'] == 'M']['Fijo'].values
            fijo_M_sig = int(float(fijo_M_sig_vals[0])) if len(fijo_M_sig_vals) > 0 else None
            
            if fijo_M_sig is not None:
                decena_N = fijo_N // 10
                unidad_M_sig = fijo_M_sig % 10
                if decena_N == unidad_M_sig:
                    eventos['N->M'].append({'fecha': fecha, 'digito': decena_N})
    
    fecha_hoy_date = fecha_hoy.date()
    stats = []
    
    for tipo, eventos_lista in eventos.items():
        if len(eventos_lista) >= 2:
            gaps = [(eventos_lista[j]['fecha'] - eventos_lista[j-1]['fecha']).days for j in range(1, len(eventos_lista))]
            promedio_historico = round(np.mean(gaps), 1) if gaps else 0
            secuencia_reciente = round(np.mean(gaps[-2:]), 1) if len(gaps) >= 2 else (gaps[0] if gaps else promedio_historico)
            
            tipo_secuencia = "NORMAL"
            prediccion_dias = promedio_historico
            
            ultimo_evento = eventos_lista[-1]
            ultima_fecha = ultimo_evento['fecha']
            ultimo_digito = ultimo_evento['digito']
            dias_sin_evento = (fecha_hoy_date - ultima_fecha).days
            
            if dias_sin_evento >= prediccion_dias:
                estado_ciclo = "ALERTA"
                alerta = True
            else:
                estado_ciclo = "EN CICLO"
                alerta = False
            
            frecuencia = len(eventos_lista)
            
        else:
            frecuencia = len(eventos_lista)
            dias_sin_evento = 999
            promedio_historico = 0
            estado_ciclo = "SIN DATOS"
            alerta = False
            ultima_fecha = None
            ultimo_digito = None
            prediccion_dias = 0
        
        stats.append({
            'Transferencia': tipo, 'Frecuencia': frecuencia,
            'Promedio_Historico': promedio_historico, 'Tipo_Secuencia': tipo_secuencia,
            'Prediccion_Dias': prediccion_dias, 'Ultima_Fecha': ultima_fecha.strftime('%d/%m/%Y') if ultima_fecha else 'N/A',
            'Ultimo_Digito': ultimo_digito if ultimo_digito is not None else 'N/A',
            'Dias_Sin_Evento': dias_sin_evento, 'Estado_Ciclo': estado_ciclo,
            'Alerta': alerta
        })
    
    return pd.DataFrame(stats)

def eliminar_ultimo_sorteo(ruta_csv):
    try:
        with open(ruta_csv, 'r', encoding='utf-8-sig') as f:
            lineas = f.readlines()
        if len(lineas) > 1:
            linea_eliminada = lineas[-1].strip()
            with open(ruta_csv, 'w', encoding='utf-8-sig') as f:
                f.writelines(lineas[:-1])
            return True, linea_eliminada
        return False, "No hay registros para eliminar"
    except Exception as e:
        return False, str(e)

# --- MAIN ---
df_fijos, df_completo = cargar_datos_geotodo(RUTA_CSV)

# SIDEBAR
st.sidebar.header("📋 Últimos Sorteos")

# Métricas para 3 sesiones
df_ultima_manana = df_fijos[df_fijos['Tipo_Sorteo'] == 'M'].tail(1)
if not df_ultima_manana.empty:
    fecha_m = df_ultima_manana['Fecha'].values[0]
    num_m = int(df_ultima_manana['Numero'].values[0])
    st.sidebar.metric("🌅 Último Mañana", f"{num_m:02d}", delta=pd.Timestamp(fecha_m).strftime('%d/%m'))

df_ultima_tarde = df_fijos[df_fijos['Tipo_Sorteo'] == 'T'].tail(1)
if not df_ultima_tarde.empty:
    fecha_t = df_ultima_tarde['Fecha'].values[0]
    num_t = int(df_ultima_tarde['Numero'].values[0])
    st.sidebar.metric("🌞 Último Tarde", f"{num_t:02d}", delta=pd.Timestamp(fecha_t).strftime('%d/%m'))

df_ultima_noche = df_fijos[df_fijos['Tipo_Sorteo'] == 'N'].tail(1)
if not df_ultima_noche.empty:
    fecha_n = df_ultima_noche['Fecha'].values[0]
    num_n = int(df_ultima_noche['Numero'].values[0])
    st.sidebar.metric("🌙 Último Noche", f"{num_n:02d}", delta=pd.Timestamp(fecha_n).strftime('%d/%m'))

with st.sidebar.expander("📝 Agregar Sorteo"):
    f = st.date_input("Fecha:", datetime.now().date(), format="DD/MM/YYYY", label_visibility="collapsed")
    s = st.radio("Sesión:", ["Mañana (M)", "Tarde (T)", "Noche (N)"], horizontal=True, label_visibility="collapsed")
    cent = st.number_input("Centena:", 0, 9, 0)
    c1, c2 = st.columns(2)
    with c1: fj = st.number_input("Fijo", 0, 99, 0, format="%02d")
    with c2: c1v = st.number_input("1er Corrido", 0, 99, 0, format="%02d")
    p2 = st.number_input("2do Corrido", 0, 99, 0, format="%02d")
    
    if st.button("💾 Guardar", type="primary"):
        cd = s.split('(')[1].replace(')', '')
        try:
            with open(RUTA_CSV, 'r', encoding='utf-8-sig') as file:
                primera = file.readline()
            sep = ',' if ',' in primera and ';' not in primera else ';'
            with open(RUTA_CSV, 'a', encoding='utf-8-sig') as file:
                file.write(f"{f.strftime('%d-%m-%Y')}{sep}{cd}{sep}{cent}{sep}{fj:02d}{sep}{c1v:02d}{sep}{p2:02d}\n")
            st.success("✅ Guardado")
            time.sleep(1)
            st.cache_resource.clear()
            st.rerun()
        except Exception as e: st.error(f"Error: {e}")

with st.sidebar.expander("🗑️ Eliminar Sorteo"):
    if st.button("❌ Eliminar Último"):
        exito, mensaje = eliminar_ultimo_sorteo(RUTA_CSV)
        if exito:
            st.success(f"✅ Eliminado: {mensaje}")
            time.sleep(1)
            st.rerun()
        else: st.error(f"❌ {mensaje}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Recargar"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.subheader("🎲 Modo")
modo = st.sidebar.radio("Filtro:", ["General", "Mañana", "Tarde", "Noche"])

if modo == "Mañana": dfa = df_fijos[df_fijos['Tipo_Sorteo'] == 'M'].copy()
elif modo == "Tarde": dfa = df_fijos[df_fijos['Tipo_Sorteo'] == 'T'].copy()
elif modo == "Noche": dfa = df_fijos[df_fijos['Tipo_Sorteo'] == 'N'].copy()
else: dfa = df_fijos.copy()

if dfa.empty:
    st.warning(f"⚠️ No hay datos para: {modo}")
    st.stop()

# Pestañas completas
tabs = st.tabs(["🗓️ Faltantes del Mes", "🔄 Transferencia", "🔢 Dígito Faltante", "🔍 Patrones", "📅 Almanaque", "🧠 Propuesta", "🔗 Secuencia", "🧪 Laboratorio", "📉 Estabilidad"])

# PESTAÑA 0: FALTANTES DEL MES
with tabs[0]:
    st.subheader("🗓️ Análisis de Faltantes del Mes")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    meses_nombres = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
                     7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    
    with col_f1:
        mes_sel = st.selectbox("Mes a Analizar:", list(meses_nombres.values()), index=datetime.now().month - 1)
        mes_num = [k for k, v in meses_nombres.items() if v == mes_sel][0]
    
    with col_f2:
        anio_sel = st.number_input("Año:", min_value=2020, max_value=datetime.now().year, value=datetime.now().year)
    
    with col_f3:
        cant_sorteos = st.slider("Sorteos para Frecuencia:", 100, 5000, 1000, step=100)

    if st.button("🔍 Analizar Faltantes", type="primary"):
        with st.spinner("Calculando..."):
            df_faltantes_res, error_msg, df_salidos_mes = analizar_faltantes_mes(dfa, mes_num, anio_sel, cant_sorteos)
        
        if error_msg:
            st.info(error_msg)
        elif not df_faltantes_res.empty:
            total_faltantes = len(df_faltantes_res)
            alta = df_faltantes_res[df_faltantes_res['Prioridad'] == '🔴 ALTA']
            
            st.markdown(f"### ⏳ Faltan por salir: {total_faltantes} números")
            st.markdown(f"#### 🔴 Prioridad Alta (Apostar): {len(alta)} números")
            st.write(" ".join([f"`{n}`" for n in alta['Número'].tolist()]))

            st.markdown("---")
            st.markdown("#### 📊 Detalle de Faltantes")
            df_show = df_faltantes_res.rename(columns={'Veces Salidas': f'Frec. ({cant_sorteos} sort.)'})
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 📝 Historial de Aciertos Recientes (Verificación)")
            st.info("Ordenado del más reciente al más antiguo. Orden: Noche (N) -> Tarde (T) -> Mañana (M).")
            
            df_historial = df_fijos.tail(30).copy()
            # Ordenamiento: N(0), T(1), M(2)
            orden_tipo = {'N': 0, 'T': 1, 'M': 2, 'OTRO': 3}
            df_historial['orden_tipo'] = df_historial['Tipo_Sorteo'].map(orden_tipo)
            df_historial = df_historial.sort_values(by=['Fecha', 'orden_tipo'], ascending=[False, True])
            df_historial['¿Salió en el Mes?'] = df_historial['Numero'].apply(lambda x: "✅ SÍ" if x in df_salidos_mes['Numero'].values else "")
            df_historial['Fecha Str'] = df_historial['Fecha'].dt.strftime('%d/%m/%Y')
            
            st.dataframe(
                df_historial[['Fecha Str', 'Tipo_Sorteo', 'Numero', '¿Salió en el Mes?']].head(15),
                column_config={
                    "Fecha Str": "Fecha", "Tipo_Sorteo": "Sorteo",
                    "Numero": st.column_config.NumberColumn("Número", format="%02d"),
                    "¿Salió en el Mes?": "Estado"
                },
                hide_index=True
            )

# PESTAÑA 1: TRANSFERENCIA
with tabs[1]:
    st.subheader("🔄 Transferencia Decena → Unidad")
    st.markdown("**Analiza cuando la decena de un sorteo pasa como unidad al siguiente**")
    st.info("M→T: Decena Mañana → Unidad Tarde | T→N: Decena Tarde → Unidad Noche | N→M: Decena Noche → Unidad Mañana (día siguiente)")
    
    dias_stats = st.slider("Días de historial:", 30, 365, 180, key="trans_stats")
    
    if st.button("Analizar Transferencias", type="primary", key="btn_trans"):
        with st.spinner("Analizando..."):
            df_stats = analizar_transferencia_geotodo(df_completo, dias_stats)
        
        for _, row in df_stats.iterrows():
            st.markdown(f"### 📊 **{row['Transferencia']}**")
            
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.metric("📅 Promedio Histórico", f"{row['Promedio_Historico']} días")
            with col_pred2:
                st.metric("⚡ Tipo Secuencia", row['Tipo_Secuencia'])
            
            if row['Alerta']:
                st.success(f"✅ **{row['Transferencia']}** - ALERTA: Puede repetir")
                st.markdown(f"📅 Último evento: {row['Ultima_Fecha']} (dígito {row['Ultimo_Digito']})")
                st.markdown(f"📊 Sin evento hace: {row['Dias_Sin_Evento']} días")
            else:
                st.info(f"⏳ **{row['Transferencia']}** - {row['Estado_Ciclo']}")
            st.markdown("---")

# PESTAÑA 2: DÍGITO FALTANTE
with tabs[2]:
    st.subheader("🔢 Análisis de Dígito Faltante")
    
    tab1, tab2, tab3 = st.tabs(["📅 Por Fecha", "📊 Estadísticas", "🧪 Backtest"])
    
    with tab1:
        st.markdown("### Selecciona una fecha para analizar")
        fecha_sel = st.date_input("Fecha:", datetime.now().date(), key="dig_falt_fecha")
        fecha_dt = datetime.combine(fecha_sel, datetime.min.time())
        
        if st.button("Analizar Fecha", key="btn_dig_falt"):
            resultado, error = analizar_dia_completo(df_completo, fecha_dt)
            
            if resultado:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Dígitos presentes")
                    st.write(resultado['digitos_presentes'])
                with col2:
                    st.subheader("❌ Dígitos faltantes")
                    if resultado['digitos_faltantes']:
                        st.warning(f"Dígitos que NO aparecieron: {resultado['digitos_faltantes']}")
                    else:
                        st.success("¡Todos los dígitos aparecieron!")
            else:
                st.error(error)
    
    with tab2:
        st.subheader("📊 Estadísticas por dígito")
        dias_stats_dig = st.slider("Días de análisis:", 30, 365, 180, key="dig_stats_dias")
        tipo_stats = st.selectbox("Ver estadísticas de:", ['general', 'centena', 'fijo', 'corrido1', 'corrido2'], key="sel_tipo_stats")
        
        stats = estadisticas_digitos_separadas(df_completo, dias_stats_dig)
        
        st.markdown(f"### Estadísticas: {tipo_stats.upper()}")
        st.dataframe(stats[tipo_stats], use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("🧪 Backtest del Dígito Faltante")
        dias_backtest = st.slider("Días para backtest:", 30, 180, 90, key="dig_backtest_dias")
        
        if st.button("Ejecutar Backtest", key="btn_dig_backtest"):
            with st.spinner("Analizando..."):
                resultado_bt = backtest_digito_faltante(df_completo, dias_backtest)
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("📊 Total evaluados", resultado_bt['total_evaluados'])
            with col2: st.metric("✅ Aciertos", resultado_bt['aciertos'])
            with col3: st.metric("🎯 Efectividad", f"{resultado_bt['efectividad']}%")

# PESTAÑA 3: PATRONES
with tabs[3]:
    st.subheader(f"🔍 Patrones")
    c1, c2 = st.columns(2)
    with c1: n = st.number_input("Disparador:", 0, 99, 40, format="%02d", key="patron_num")
    with c2: v = st.slider("Ventana:", 1, 30, 15, key="patron_ventana")
    
    if st.button("Analizar", key="b1"):
        st.session_state['sb1'] = True
    
    if st.session_state.get('sb1'):
        r, tot = analizar_siguientes(dfa, n, v)
        if r is None:
            st.warning(f"⚠️ El número {n:02d} no ha salido.")
        else:
            st.success(f"📊 Encontrado {tot} veces.")
            st.dataframe(r.head(20), hide_index=True)

# PESTAÑA 4: ALMANAQUE
with tabs[4]:
    st.subheader("📅 Almanaque")
    
    with st.form("almanaque_form"):
        c_r, c_m = st.columns(2)
        with c_r:
            ca, cb = st.columns(2)
            with ca: dia_inicio = st.number_input("Día Ini:", 1, 31, 16)
            with cb: dia_fin = st.number_input("Día Fin:", 1, 31, 20)
        with c_m: meses_atras = st.slider("Meses Atrás:", 1, 12, 4)
        
        submitted = st.form_submit_button("📊 Analizar", type="primary")
        
        if submitted:
            if dia_inicio > dia_fin:
                st.error("❌ El día de inicio no puede ser mayor al final.")
            else:
                with st.spinner("Analizando..."):
                    res = analizar_almanaque(dfa, int(dia_inicio), int(dia_fin), int(meses_atras), strict_mode=False)
                
                if not res['success']:
                    st.error(f"❌ {res.get('mensaje', 'Error')}")
                else:
                    if res['nombres_bloques']:
                        st.success(f"📅 Periodos: {', '.join(res['nombres_bloques'])}")
                    
                    st.markdown("---")
                    st.subheader("⏱️ Evaluación en Tiempo Real")
                    st.info(f"**Estado:** {res['estado_periodo']}")
                    
                    col_h, col_f = st.columns([2, 1])
                    with col_h:
                        if not res['df_historial_actual'].empty:
                            hist_view = res['df_historial_actual'].copy()
                            hist_view['Fecha'] = hist_view['Fecha'].dt.strftime('%d/%m/%Y')
                            st.markdown("### 📜 Resultados del Mes")
                            st.dataframe(hist_view, use_container_width=True, hide_index=True)
                        else:
                            st.info("No hay resultados aún.")
                    
                    with col_f:
                        st.markdown("### ⏳ Faltantes")
                        if not res['df_faltantes'].empty:
                            st.dataframe(res['df_faltantes'], use_container_width=True, hide_index=True)
                        else:
                            st.success("🎉 ¡Todos salieron!")
                    
                    st.markdown("---")
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.markdown("### 🔢 Decenas")
                        st.dataframe(res['df_dec'], hide_index=True)
                    with col_d2:
                        st.markdown("### 🔢 Unidades")
                        st.dataframe(res['df_uni'], hide_index=True)
                    
                    st.markdown("---")
                    col_t1, col_t2 = st.columns([1, 2])
                    with col_t1:
                        st.markdown("### 🔥 Tendencia")
                        if not res['tend'].empty:
                            mv = int(res['tend']['Frecuencia'].max())
                            st.dataframe(res['tend'], column_config={
                                "Frecuencia": st.column_config.ProgressColumn("Frecuencia", format="%d", min_value=0, max_value=mv)
                            }, hide_index=True)
                            st.info(f"Dominante: **{res['top_p']}**")
                    
                    with col_t2:
                        st.markdown("### 💡 Sugerencias")
                        if not res['df_tend_nums'].empty:
                            st.dataframe(res['df_tend_nums'], hide_index=True)
                    
                    with st.expander("🛡️ Persistencia"):
                        p1, p2 = st.columns(2)
                        with p1:
                            st.markdown("#### 📌 Números")
                            if not res['df_pers_num'].empty:
                                st.dataframe(res['df_pers_num'], hide_index=True)
                            else:
                                st.info("Ninguno.")
                        with p2:
                            st.markdown("#### 🏷️ Perfiles")
                            if res['persistentes_perfiles']:
                                st.dataframe(pd.DataFrame(list(res['persistentes_perfiles']), columns=["Perfil"]), hide_index=True)
                            else:
                                st.info("Ninguno.")
                    
                    with st.expander("📋 Ranking"):
                        if not res['df_rank'].empty:
                            st.dataframe(res['df_rank'].head(20), hide_index=True)

# PESTAÑA 5: PROPUESTA
with tabs[5]:
    st.subheader(f"🧠 Sincronización")
    c1, c2 = st.columns(2)
    with c1: dt = st.number_input("Días Tendencia:", 5, 60, 15, key="prop_dias")
    with c2: dg = st.number_input("Gap Mínimo:", 1, 90, 10, key="prop_gap")
    
    if st.button("🧠 Generar", key="b_pr"):
        st.session_state['spr'] = True
    
    if st.session_state.get('spr'):
        p = generar_sugerencia(dfa, dt, dg)
        if p.empty:
            st.warning("No hay sugerencias.")
        else:
            st.dataframe(p, hide_index=True)

# PESTAÑA 6: SECUENCIA
with tabs[6]:
    st.subheader(f"🔗 Secuencia")
    c1, c2, c3 = st.columns(3)
    with c1: parte = st.selectbox("Parte del número", ["Decena", "Unidad"])
    with c2: tipo = st.selectbox("Tipo de patrón", ["digito", "paridad", "altura"])
    with c3: secuencia = st.text_input("Secuencia:", "0 1 2")
    
    if st.button("Buscar"):
        r, e = buscar_seq(dfa, parte, tipo, secuencia)
        if r is not None:
            st.success(f"Se encontraron {len(r)} coincidencias")
            st.dataframe(r, use_container_width=True, hide_index=True)
        else:
            st.error(e)
    
    st.markdown("""
    **Ayuda:**
    - **Dígito**: Usa valores 0-9
    - **Paridad**: P (par) o I (impar)
    - **Altura**: A (alto: 5-9) o B (bajo: 0-4)
    """)

# PESTAÑA 7: LABORATORIO
with tabs[7]:
    st.subheader("🧪 Simulador")
    
    meses_lab = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
                 7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    fecha_hoy_lab = datetime.now()
    mes_default_lab = fecha_hoy_lab.month - 1 if fecha_hoy_lab.month > 1 else 12
    anio_default_lab = fecha_hoy_lab.year if fecha_hoy_lab.month > 1 else fecha_hoy_lab.year - 1
    
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        nombre_mes_sel = st.selectbox("Mes:", list(meses_lab.values()), index=list(meses_lab.keys()).index(mes_default_lab), key="lab_mes")
        mes_sel_num = [k for k, v in meses_lab.items() if v == nombre_mes_sel][0]
    with col_l2:
        anio_sel = st.number_input("Año:", min_value=2020, max_value=2030, value=anio_default_lab, key="lab_anio")
    with col_l3:
        c_dia1, c_dia2 = st.columns(2)
        with c_dia1: dia_ini_lab = st.number_input("Día Ini:", 1, 31, 1, key="lab_dia_ini")
        with c_dia2: dia_fin_lab = st.number_input("Día Fin:", 1, 31, 15, key="lab_dia_fin")
    
    meses_atras_sim = st.slider("Meses atrás:", 2, 6, 3, key="lab_meses_atras")

    if st.button("🚀 Ejecutar", type="primary"):
        with st.spinner("Analizando..."):
            res = backtesting_estrategia(dfa, mes_sel_num, anio_sel, dia_ini_lab, dia_fin_lab, meses_atras_sim)
            
            if res is not None:
                st.success(f"✅ Efectividad: {res['Efectividad']}%")
                
                col_izq, col_der = st.columns(2)
                
                with col_izq:
                    st.markdown("### 📋 Estrategia")
                    st.caption(f"Basada en: {res['Entrenamiento']}")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🔥 Calientes D", str(res['hot_d']))
                    c2.metric("🟡 Tibios D", str(res['warm_d']))
                    c3.metric("🧊 Fríos D", str(res['cold_d']))
                    
                    c4, c5, c6 = st.columns(3)
                    c4.metric("🔥 Calientes U", str(res['hot_u']))
                    c5.metric("🟡 Tibios U", str(res['warm_u']))
                    c6.metric("🧊 Fríos U", str(res['cold_u']))
                    
                    st.markdown("**Perfiles Persistentes:**")
                    for p in res['Perfiles']:
                        st.markdown(f"- 🏷️ {p}")

                with col_der:
                    st.markdown("### 🎲 Resultados")
                    st.caption(f"Periodo: {res['Prueba']}")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total", res['Total'])
                    m2.metric("Aciertos", res['Aciertos'])
                    m3.metric("Sufrientes", res['Sufrientes'])
                    
                    st.markdown("#### Detalle:")
                    df_view = res['Detalle'].copy()
                    df_view['Fecha'] = df_view['Fecha'].dt.strftime('%d/%m/%Y')
                    st.dataframe(df_view, use_container_width=True, hide_index=True)
            else:
                st.error(f"🛑 Error en el análisis")

# PESTAÑA 8: ESTABILIDAD
with tabs[8]:
    st.subheader("📉 Estabilidad")
    
    dias_analisis = st.slider("Días de Historial:", 90, 3650, 365, step=30, key="est_dias")
    
    if st.button("📊 Calcular", key="b_est"):
        with st.spinner("Analizando..."):
            df_est = analizar_estabilidad_numeros(dfa, dias_analisis)
            
            if df_est is None:
                st.error("Sin datos suficientes.")
            else:
                st.markdown("### 🏆 Ranking")
                
                st.dataframe(
                    df_est.head(30),
                    column_config={
                        "Estado": st.column_config.TextColumn("Estado"),
                        "Gap Actual": st.column_config.NumberColumn("Días sin salir", format="%d"),
                        "Gap Máximo (Días)": st.column_config.NumberColumn("Max", format="%d"),
                        "Gap Promedio": st.column_config.NumberColumn("Prom", format="%.1f"),
                        "Desviación (Irregularidad)": st.column_config.NumberColumn("Irreg", format="%.1f"),
                        "Última Salida": st.column_config.TextColumn("Último")
                    },
                    hide_index=True
                )
                
                st.info("💡 **Estados:** 🔥 EN RACHA | ✅ NORMAL | ⏳ VENCIDO | 🔴 MUY VENCIDO")

st.markdown("---")
st.caption("Geotodo Suite Ultimate v1.1 | Adaptado para 3 Sesiones (M, T, N)")