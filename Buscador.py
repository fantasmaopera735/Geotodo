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

# --- CONFIGURACIÓN DE LA RUTA RELATIVA ---
RUTA_CSV = 'Geotodo.csv' 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Geotodo - Suite Ultimate",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Geotodo - Suite Ultimate")

# --- FUNCIÓN PARA CARGAR DATOS ---
@st.cache_resource
def cargar_datos_geotodo(_ruta_csv):
    try:
        if not os.path.exists(_ruta_csv):
            st.error(f"❌ Error: No se encontró el archivo {_ruta_csv}.")
            st.info("💡 Asegúrate de que el archivo CSV esté en el mismo directorio que este script.")
            st.stop()
        
        df = pd.read_csv(_ruta_csv, sep=';', encoding='latin-1')
        
        # Verificar columnas necesarias
        columnas_requeridas = ['Fecha']
        if not all(col in df.columns for col in columnas_requeridas):
            if 'Fecha' not in df.columns and len(df.columns) > 0:
                df.columns = ['Fecha', 'Tipo_Sorteo', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido'] + list(df.columns[5:])
        
        # Renombrar columnas si es necesario
        rename_map = {}
        cols = df.columns.tolist()
        for col in cols:
            col_lower = col.lower().strip()
            if 'fecha' in col_lower:
                rename_map[col] = 'Fecha'
            elif 'tarde' in col_lower or 'noche' in col_lower or 'sorteo' in col_lower:
                rename_map[col] = 'Tipo_Sorteo'
            elif col_lower == 'fijo':
                rename_map[col] = 'Fijo'
            elif 'corrido' in col_lower and '1' in col_lower:
                rename_map[col] = 'Primer_Corrido'
            elif 'corrido' in col_lower and '2' in col_lower:
                rename_map[col] = 'Segundo_Corrido'
        
        df.rename(columns=rename_map, inplace=True)
        
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)
        
        if 'Tipo_Sorteo' in df.columns:
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper()
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].map({
                'MAÑANA': 'M', 'M': 'M', 'MANANA': 'M',
                'TARDE': 'T', 'T': 'T',
                'NOCHE': 'N', 'N': 'N'
            }).fillna('OTRO')
        else:
            df['Tipo_Sorteo'] = 'OTRO'
        
        if 'Fijo' in df.columns:
            df_fijos = df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy()
            df_fijos = df_fijos.rename(columns={'Fijo': 'Numero'})
            df_fijos['Numero'] = pd.to_numeric(df_fijos['Numero'], errors='coerce')
            df_fijos = df_fijos.dropna(subset=['Numero'])
            df_fijos['Numero'] = df_fijos['Numero'].astype(int)
        else:
            st.error("❌ No se encontró la columna 'Fijo' en el archivo CSV.")
            st.stop()
        
        draw_order_map = {'M': 0, 'T': 1, 'N': 2, 'OTRO': 3}
        df_fijos['draw_order'] = df_fijos['Tipo_Sorteo'].map(draw_order_map).fillna(3)
        df_fijos['sort_key'] = df_fijos['Fecha'] + pd.to_timedelta(df_fijos['draw_order'], unit='h')
        df_fijos = df_fijos.sort_values(by='sort_key').reset_index(drop=True)
        
        return df_fijos
        
    except pd.errors.EmptyDataError:
        st.error("❌ El archivo CSV está vacío.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        st.stop()

# --- FUNCIÓN 1: PATRONES ---
def analizar_siguientes(df_fijos, numero_busqueda, ventana_sorteos):
    indices = df_fijos[df_fijos['Numero'] == numero_busqueda].index.tolist()
    if not indices: 
        return None, 0 
    
    lista_s = []
    for idx in indices:
        i = idx + 1
        f = idx + ventana_sorteos + 1
        if i < len(df_fijos): 
            lista_s.extend(df_fijos.iloc[i:f]['Numero'].tolist())
    
    if not lista_s:
        return None, len(indices)
    
    c = Counter(lista_s)
    r = pd.DataFrame.from_dict(c, orient='index', columns=['Frecuencia'])
    r['Probabilidad (%)'] = (r['Frecuencia'] / len(lista_s) * 100).round(2)
    r['Número'] = [f"{int(x):02d}" for x in r.index]
    return r.sort_values('Frecuencia', ascending=False), len(indices)

# --- FUNCIÓN 2: ALMANAQUE (CORREGIDO - ORDEN N-T-M) ---
def analizar_almanaque(df_fijos, dia_inicio, dia_fin, meses_atras, strict_mode=True):
    fecha_hoy = datetime.now()
    
    bloques_validos = []
    nombres_bloques = []
    debug_info = []
    
    for offset in range(1, meses_atras + 1):
        f_obj = fecha_hoy - relativedelta(months=offset)
        try:
            last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
            f_i = datetime(f_obj.year, f_obj.month, min(dia_inicio, last_day))
            f_f = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
            
            if f_i > f_f: 
                continue
                
            df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
            
            debug_info.append(f"{f_obj.strftime('%B %Y')}: {len(df_b)} registros en rango {dia_inicio}-{dia_fin}.")
            
            if not df_b.empty:
                bloques_validos.append(df_b)
                nombres_bloques.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
                
        except Exception as e:
            debug_info.append(f"Error en {f_obj.strftime('%B %Y')}: {str(e)}")
            continue

    mensaje_advertencia = ""
    fallback_usado = False
    
    if not bloques_validos:
        if strict_mode:
            return {
                'success': False,
                'mensaje': "Sin datos en modo estricto.",
                'debug_info': debug_info
            }
        
        debug_info.append("Modo Fallback activado.")
        
        for offset in range(1, meses_atras + 1):
            f_obj = fecha_hoy - relativedelta(months=offset)
            try:
                f_i = datetime(f_obj.year, f_obj.month, 1)
                last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
                f_f = datetime(f_obj.year, f_obj.month, last_day)
                
                df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
                
                debug_info.append(f"FALLBACK {f_obj.strftime('%B %Y')}: {len(df_b)} registros.")
                
                if not df_b.empty:
                    bloques_validos.append(df_b)
                    nombres_bloques.append(f"{f_obj.strftime('%b')} (Todo el mes)")
                    
            except Exception as e:
                debug_info.append(f"Error Fallback: {str(e)}")
                continue
        
        if not bloques_validos:
            return {
                'success': False,
                'mensaje': "Sin datos.",
                'debug_info': debug_info
            }
        
        fallback_usado = True
        mensaje_advertencia = "(Usando datos del mes completo)."
    
    # Procesar datos
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
        p = f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}"
        ranking.append({'Número': f"{n:02d}", 'Frecuencia': v, 'Perfil': p})
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
            ed, eu = mapa_d.get(d, '?'), mapa_u.get(u, '?')
            perfiles_en_bloque.add(f"{ed} + {eu}")
        sets_perfiles.append(perfiles_en_bloque)
    
    persistentes_perfiles = set.intersection(*sets_perfiles) if sets_perfiles else set()
    persistentes_num_set = set(p['Número'] for p in pers_num) if pers_num else set()

    # Evaluar mes actual - CORREGIDO ORDEN N-T-M
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
                    ed, eu = mapa_d.get(d, '?'), mapa_u.get(u, '?')
                    perfil_completo = f"{ed} + {eu}"
                    
                    cumple_regla = False
                    motivo = ""
                    
                    if f"{num:02d}" in persistentes_num_set:
                        cumple_regla = True
                        motivo = "Num. Persistente"
                    elif perfil_completo in persistentes_perfiles:
                        cumple_regla = True
                        motivo = "Perfil Persistente"
                    
                    historial_data.append({
                        'Fecha': row.Fecha,
                        'Tipo_Sorteo': row.Tipo_Sorteo,
                        'Número': f"{num:02d}",
                        'Perfil (D/U)': perfil_completo,
                        'Cumple Regla': '✅ SÍ' if cumple_regla else '❌ NO',
                        'Tipo Regla': motivo if cumple_regla else '-'
                    })
                
                df_historial_actual = pd.DataFrame(historial_data)
                
                # ORDENAR: N (más reciente) > T > M (menos reciente)
                orden_sorteo = {'N': 0, 'T': 1, 'M': 2, 'OTRO': 3}
                df_historial_actual['orden'] = df_historial_actual['Tipo_Sorteo'].map(orden_sorteo).fillna(3)
                df_historial_actual = df_historial_actual.sort_values(
                    ['Fecha', 'orden'], ascending=[False, True]
                ).reset_index(drop=True)
                df_historial_actual = df_historial_actual.drop(columns=['orden'])
            
            estado_periodo = f"🟢 PERIODO ACTIVO (Evaluado hasta: {hoy.strftime('%d/%m')})"
            
    except Exception as e:
        estado_periodo = f"⚪ Error en fechas: {str(e)}"
        df_historial_actual = pd.DataFrame()

    # Calcular faltantes
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
            df_faltantes = pd.DataFrame([{'Número': n, 'Estado': '⏳ FALTANTE'} 
                                         for n in sorted(list(faltantes_nums))])

    return {
        'success': True,
        'df_total': df_total,
        'df_dec': df_dec,
        'df_uni': df_uni,
        'df_3x3': df_3x3,
        'df_rank': df_rank,
        'nombres_bloques': nombres_bloques,
        'df_pers_num': df_pers_num,
        'tend': tend,
        'top_p': top_p,
        'df_tend_nums': df_tend_nums,
        'persistentes_perfiles': persistentes_perfiles,
        'df_historial_actual': df_historial_actual,
        'df_faltantes': df_faltantes,
        'estado_periodo': estado_periodo,
        'mensaje_advertencia': mensaje_advertencia,
        'debug_info': debug_info,
        'fallback_usado': fallback_usado
    }

# --- FUNCIÓN 5: BACKTESTING ---
def backtesting_estrategia_congelada(df_fijos, mes_objetivo, anio_objetivo, dia_ini, dia_fin, meses_atras):
    try:
        fecha_ref = datetime(anio_objetivo, mes_objetivo, 1)
        bloques_train = []
        nombres_bloques_train = []
        
        for offset in range(1, meses_atras + 1):
            f_obj = fecha_ref - relativedelta(months=offset)
            try:
                last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
                f_i = datetime(f_obj.year, f_obj.month, min(dia_ini, last_day))
                f_f = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
                
                if f_i > f_f: 
                    continue
                
                df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
                
                if not df_b.empty:
                    bloques_train.append(df_b)
                    nombres_bloques_train.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
                else:
                    nombres_bloques_train.append(f"({f_obj.strftime('%B %Y')} Vacio)")
                    
            except Exception as e:
                nombres_bloques_train.append(f"Error: {str(e)}")

        if not bloques_train:
            return None, "Sin datos para entrenamiento."

        df_train_total = pd.concat(bloques_train)
        df_train_total['Dec'] = df_train_total['Numero'] // 10
        df_train_total['Uni'] = df_train_total['Numero'] % 10
        cnt_d = df_train_total['Dec'].value_counts().reindex(range(10), fill_value=0)
        cnt_u = df_train_total['Uni'].value_counts().reindex(range(10), fill_value=0)
        
        def get_lists(serie):
            df_t = serie.sort_values(ascending=False).reset_index()
            df_t.columns = ['Digito', 'Frecuencia']
            conds = [(df_t.index < 3), (df_t.index < 6)]
            vals = ['🔥 Caliente', '🟡 Tibio']
            df_t['Estado'] = np.select(conds, vals, default='🧊 Frío')
            mapa = {r['Digito']: r['Estado'] for _, r in df_t.iterrows()}
            hot = [r['Digito'] for _, r in df_t.iterrows() if r['Estado'] == '🔥 Caliente']
            warm = [r['Digito'] for _, r in df_t.iterrows() if r['Estado'] == '🟡 Tibio']
            cold = [r['Digito'] for _, r in df_t.iterrows() if r['Estado'] == '🧊 Frío']
            return mapa, hot, warm, cold

        mapa_d_congelado, hot_d_list, warm_d_list, cold_d_list = get_lists(cnt_d)
        mapa_u_congelado, hot_u_list, warm_u_list, cold_u_list = get_lists(cnt_u)
        
        sets_perfiles = []
        for df_b in bloques_train:
            perfiles_bloque = set()
            for row in df_b.itertuples():
                d, u = row.Numero // 10, row.Numero % 10
                perfiles_bloque.add(f"{mapa_d_congelado.get(d, '?')} + {mapa_u_congelado.get(u, '?')}")
            sets_perfiles.append(perfiles_bloque)
        perfiles_persistentes = set.intersection(*sets_perfiles) if sets_perfiles else set()

        f_prueba_ini = datetime(anio_objetivo, mes_objetivo, min(dia_ini, 28))
        f_prueba_fin = datetime(anio_objetivo, mes_objetivo, min(dia_fin, calendar.monthrange(anio_objetivo, mes_objetivo)[1]))
        df_test = df_fijos[(df_fijos['Fecha'] >= f_prueba_ini) & (df_fijos['Fecha'] <= f_prueba_fin)]
        
        if df_test.empty:
            return None, "Sin datos en el mes de prueba."

        resultados_detalle = []
        aciertos_persistencia = 0
        sufrientes_count = 0
        
        for row in df_test.itertuples():
            num = row.Numero
            d, u = num // 10, num % 10
            perfil = f"{mapa_d_congelado.get(d, '?')} + {mapa_u_congelado.get(u, '?')}"
            
            es_pers = perfil in perfiles_persistentes
            if es_pers: 
                aciertos_persistencia += 1
            
            es_sufriente = (d in cold_d_list) or (u in cold_u_list)
            if es_sufriente: 
                sufrientes_count += 1
            
            estado_d = mapa_d_congelado.get(d, '?')
            estado_u = mapa_u_congelado.get(u, '?')
            
            resultados_detalle.append({
                'Fecha': row.Fecha,
                'Número': num,
                'Decena (Congelada)': f"{d} ({estado_d})",
                'Unidad (Congelada)': f"{u} ({estado_u})",
                'Resultado': '✅ ESTRUCTURA' if es_pers else ('⚡ SUFRIENTE' if es_sufriente else '❌ OTRO')
            })
            
        df_detalle = pd.DataFrame(resultados_detalle)
        total_test = len(df_detalle)
        porc_pers = (aciertos_persistencia / total_test * 100) if total_test > 0 else 0

        return {
            "Periodo Entrenamiento": ", ".join([x for x in nombres_bloques_train if "Vacio" not in x]),
            "Periodo Prueba": f"{f_prueba_ini.strftime('%B %Y')} ({dia_ini}-{dia_fin})",
            "Perfiles Persistentes": perfiles_persistentes,
            "Total Prueba": total_test,
            "Aciertos Persistente": aciertos_persistencia,
            "% Efectividad Estructura": round(porc_pers, 2),
            "Sufrientes Exitosos": sufrientes_count,
            "Detalle": df_detalle,
            "hot_d": hot_d_list, "warm_d": warm_d_list, "cold_d": cold_d_list,
            "hot_u": hot_u_list, "warm_u": warm_u_list, "cold_u": cold_u_list
        }

    except Exception as e:
        return None, f"Error: {str(e)}"

# --- FUNCIÓN 6: ESTABILIDAD ---
def analizar_estabilidad_numeros(df_fijos, dias_analisis=365):
    fecha_limite = datetime.now() - timedelta(days=dias_analisis)
    df_historico = df_fijos[df_fijos['Fecha'] >= fecha_limite].copy()
    
    if df_historico.empty: 
        return None, "Sin datos."
    
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
            
            if gap_actual > max_gap:
                max_gap = gap_actual
            
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
            'Número': f"{num:02d}",
            'Gap Actual': gap_actual,
            'Gap Máximo (Días)': max_gap,
            'Gap Promedio': round(avg_gap, 1),
            'Desviación (Irregularidad)': round(std_gap, 1),
            'Estado': estado,
            'Última Salida': ultima_fecha.strftime('%d/%m/%Y') if ultima_fecha else "N/A"
        })
        
    df_est = pd.DataFrame(estabilidad_data)
    df_est = df_est.sort_values(by=['Gap Máximo (Días)', 'Desviación (Irregularidad)'], ascending=[True, True]).reset_index(drop=True)
    return df_est

# --- FUNCIONES AUXILIARES ---
def generar_sugerencia(df, dias, gap):
    fh = datetime.now()
    df_t = df[df['Fecha'] >= fh - timedelta(days=dias)].copy()
    if df_t.empty: 
        return pd.DataFrame()
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
                res.append({'Número': f"{n:02d}", 'Gap': g, 'Estado': "⚡ Muy" if g > gap*1.5 else "✅ Op"})
    return pd.DataFrame(res).sort_values('Gap', ascending=False) if res else pd.DataFrame()

# --- FUNCIÓN DE BÚSQUEDA DE SECUENCIA ---
def buscar_seq(df_fijos, part, type_, seq):
    try:
        p = [x.strip().upper() for x in seq.replace(',', ' ').split() if x.strip()]
    except: 
        return None, "Error al procesar secuencia."
    
    if len(p) == 0 or len(p) > 5: 
        return None, "Secuencia inválida (1-5 elementos)."
    
    if type_ == 'digito': 
        v = set(range(10))
    elif type_ == 'paridad': 
        v = {'P', 'I'}
    elif type_ == 'altura': 
        v = {'A', 'B'}
    else: 
        return None, "Tipo desconocido."
    
    try:
        if type_ == 'digito':
            p = [int(x) for x in p]
            if any(x not in v for x in p): 
                return None, "Dígitos deben ser 0-9."
        else:
            if any(x not in v for x in p): 
                return None, f"Usa valores: {', '.join(v)}."
    except: 
        return None, "Error de conversión."
    
    l = []
    for x in df_fijos['Numero']:
        val = x // 10 if part == 'Decena' else x % 10
        if type_ == 'digito':
            l.append(val)
        elif type_ == 'paridad':
            l.append('P' if val % 2 == 0 else 'I')
        elif type_ == 'altura':
            l.append('A' if val >= 5 else 'B')
        
    lp = len(p)
    dat = {}
    
    for i in range(len(l) - lp):
        if l[i:i+lp] == p:
            sig = l[i + lp]
            r = df_fijos.iloc[i + lp]
            e = f"{r['Numero']:02d} ({r['Fecha'].strftime('%d/%m/%Y')})"
            if sig not in dat: 
                dat[sig] = {'c': 0, 'e': []}
            
            dat[sig]['c'] += 1
            if len(dat[sig]['e']) < 3 and e not in dat[sig]['e']: 
                dat[sig]['e'].append(e)
    
    if not dat: 
        return None, "No se encontraron coincidencias."
    
    total_count = sum(v['c'] for v in dat.values())
    rows = []
    for k, v in dat.items():
        prob = (v['c'] / total_count * 100) if total_count > 0 else 0.0
        rows.append({
            'Siguiente': k, 
            'Frecuencia': v['c'], 
            'Ejemplos': ", ".join(v['e']), 
            'Prob': round(prob, 2)
        })
        
    df_r = pd.DataFrame(rows)
    return df_r.sort_values('Frecuencia', ascending=False), None

# --- MAIN ---
def main():
    df = cargar_datos_geotodo(RUTA_CSV)
    
    if df.empty:
        st.error("❌ El DataFrame está vacío después de cargar.")
        st.stop()
    
    meses = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
             7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    fecha_hoy = datetime.now()
    mes_default = fecha_hoy.month - 1 if fecha_hoy.month > 1 else 12
    anio_default = fecha_hoy.year if fecha_hoy.month > 1 else fecha_hoy.year - 1

    st.sidebar.header("⚙️ Panel")
    
    with st.sidebar.expander("📂 Datos", expanded=True):
        def mostrar_ultimo(l, i, ic):
            if not l.empty:
                f = l['Fecha'].max()
                n = l[l['Fecha'] == f]['Numero'].values[0]
                st.metric(f"{ic} {i}", f"{f.strftime('%d/%m')}", delta=f"{n:02d}")
        
        mostrar_ultimo(df[df['Tipo_Sorteo'] == 'M'], "Mañana", "🌅")
        mostrar_ultimo(df[df['Tipo_Sorteo'] == 'T'], "Tarde", "🌞")
        mostrar_ultimo(df[df['Tipo_Sorteo'] == 'N'], "Noche", "🌙")

    with st.sidebar.expander("📝 Agregar", expanded=False):
        f = st.date_input("Fecha:", datetime.now().date(), format="DD/MM/YYYY", label_visibility="collapsed")
        s = st.radio("Sesión:", ["Mañana (M)", "Tarde (T)", "Noche (N)"], horizontal=True, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1: 
            fj = st.number_input("Fijo", 0, 99, 0, format="%02d", label_visibility="collapsed")
        with c2: 
            c1v = st.number_input("1er", 0, 99, 0, format="%02d", label_visibility="collapsed")
        p2 = st.number_input("2do", 0, 99, 0, format="%02d", label_visibility="collapsed")
        
        if st.button("💾 Guardar", type="primary", use_container_width=True):
            cd = s.split('(')[1].replace(')', '')
            try:
                with open(RUTA_CSV, 'a', encoding='latin-1') as file:
                    file.write(f"{f.strftime('%d/%m/%Y')};{cd};{fj};{c1v};{p2}\n")
                st.success("✅ Guardado")
                time.sleep(1)
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Recargar"):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.subheader("🎲 Modo")
    modo = st.sidebar.radio("Filtro:", ["General", "Mañana", "Tarde", "Noche"])
    
    if modo == "Mañana":
        dfa = df[df['Tipo_Sorteo'] == 'M'].copy()
        t = "Mañana"
    elif modo == "Tarde":
        dfa = df[df['Tipo_Sorteo'] == 'T'].copy()
        t = "Tarde"
    elif modo == "Noche":
        dfa = df[df['Tipo_Sorteo'] == 'N'].copy()
        t = "Noche"
    else:
        dfa = df.copy()
        t = "General"
    
    if dfa.empty:
        st.warning(f"⚠️ No hay datos para: {t}")
        st.stop()
    
    tabs = st.tabs(["🔍 Patrones", "📅 Almanaque", "🧠 Propuesta", "🔗 Secuencia", "🧪 Laboratorio", "📉 Estabilidad"])

    # PESTAÑA 0: PATRONES
    with tabs[0]:
        st.subheader(f"🔍 Patrones: {t}")
        c1, c2 = st.columns(2)
        with c1: 
            n = st.number_input("Disparador:", 0, 99, 40, format="%02d", key="patron_num")
        with c2: 
            v = st.slider("Ventana:", 1, 30, 15, key="patron_ventana")
        
        if st.button("🔍 Analizar", key="b1"):
            st.session_state['sb1'] = True
        
        if st.session_state.get('sb1'):
            r, tot = analizar_siguientes(dfa, n, v)
            if r is None:
                st.warning(f"⚠️ El número {n:02d} no ha salido.")
            else:
                st.success(f"📊 Encontrado {tot} veces.")
                max_val = int(r['Frecuencia'].max()) if not r.empty else 1
                st.dataframe(r.head(20), column_config={
                    "Frecuencia": st.column_config.ProgressColumn("Frecuencia", format="%d", min_value=0, max_value=max_val)
                }, hide_index=True)

    # PESTAÑA 1: ALMANAQUE (CORREGIDO)
    with tabs[1]:
        st.subheader(f"📅 Almanaque: {t}")
        
        with st.form("almanaque_form"):
            c_r, c_m = st.columns(2)
            with c_r:
                ca, cb = st.columns(2)
                with ca: 
                    di = st.number_input("Día Ini:", 1, 31, 16, key="alm_ini")
                with cb: 
                    dfi = st.number_input("Día Fin:", 1, 31, 20, key="alm_fin")
            with c_m: 
                ma = st.slider("Meses Atrás:", 1, 12, 4, key="alm_meses")
            
            submitted = st.form_submit_button("📊 Analizar", type="primary")
            
            if submitted:
                if di > dfi:
                    st.error("❌ El día de inicio no puede ser mayor al final.")
                else:
                    with st.spinner("Analizando..."):
                        res = analizar_almanaque(dfa, int(di), int(dfi), int(ma), strict_mode=False)
                    
                    if not res['success']:
                        st.error(f"❌ {res['mensaje']}")
                        with st.expander("🔍 Detalles técnicos"):
                            for log in res['debug_info']:
                                st.text(log)
                    else:
                        if res['mensaje_advertencia']:
                            st.warning(res['mensaje_advertencia'])
                        
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
                                # Mostrar tipo de sorteo
                                hist_view['Sorteo'] = hist_view['Tipo_Sorteo']
                                hist_view = hist_view[['Fecha', 'Sorteo', 'Número', 'Perfil (D/U)', 'Cumple Regla', 'Tipo Regla']]
                                st.markdown("### 📜 Resultados del Mes (Orden: N-T-M)")
                                st.caption("Ordenado de más reciente (Noche) a menos reciente (Mañana)")
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

    # PESTAÑA 2: PROPUESTA
    with tabs[2]:
        st.subheader(f"🧠 Sincronización: {t}")
        c1, c2 = st.columns(2)
        with c1: 
            dt = st.number_input("Días Tendencia:", 5, 60, 15, key="prop_dias")
        with c2: 
            dg = st.number_input("Gap Mínimo:", 1, 90, 10, key="prop_gap")
        
        if st.button("🧠 Generar", key="b_pr"):
            st.session_state['spr'] = True
        
        if st.session_state.get('spr'):
            p = generar_sugerencia(dfa, dt, dg)
            if p.empty:
                st.warning("No hay sugerencias.")
            else:
                st.dataframe(p, hide_index=True)

    # PESTAÑA 3: SECUENCIA
    with tabs[3]:
        st.subheader(f"🔗 Secuencia: {t}")
        c1, c2, c3 = st.columns(3)
        with c1: 
            part = st.selectbox("Parte:", ["Decena", "Unidad"], key="seq_part")
        with c2: 
            type_ = st.selectbox("Tipo:", ["Digito (0-9)", "Paridad (P/I)", "Altura (A/B)"], key="seq_type")
        with c3: 
            seq = st.text_input("Secuencia:", placeholder="Ej: 1 2 3", key="seq_input")
        
        if st.button("🔗 Buscar", key="b_seq"):
            st.session_state['sseq'] = True
        
        if st.session_state.get('sseq') and seq:
            type_clean = type_.lower().replace('(', '').replace(')', '').split(' ')[0]
            r, e = buscar_seq(dfa, part, type_clean, seq)
            if e:
                st.warning(f"⚠️ {e}")
            else:
                max_val = int(r['Frecuencia'].max()) if not r.empty else 1
                st.dataframe(r, column_config={
                    "Siguiente": st.column_config.TextColumn("Sig"),
                    "Frecuencia": st.column_config.ProgressColumn("Frec", format="%d", min_value=0, max_value=max_val),
                    "Prob": st.column_config.NumberColumn("Prob", format="%.2f%%"),
                    "Ejemplos": st.column_config.TextColumn("Historial")
                }, hide_index=True)

    # PESTAÑA 4: LABORATORIO
    with tabs[4]:
        st.subheader("🧪 Simulador")
        
        meses_lab = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
                     7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
        fecha_hoy_lab = datetime.now()
        mes_default_lab = fecha_hoy_lab.month - 1 if fecha_hoy_lab.month > 1 else 12
        anio_default_lab = fecha_hoy_lab.year if fecha_hoy_lab.month > 1 else fecha_hoy_lab.year - 1
        
        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            nombre_mes_sel = st.selectbox("Mes:", list(meses_lab.values()), 
                                          index=list(meses_lab.keys()).index(mes_default_lab), key="lab_mes")
            mes_sel_num = [k for k, v in meses_lab.items() if v == nombre_mes_sel][0]
        with col_l2:
            anio_sel = st.number_input("Año:", min_value=2020, max_value=2030, value=anio_default_lab, key="lab_anio")
        with col_l3:
            c_dia1, c_dia2 = st.columns(2)
            with c_dia1: 
                dia_ini = st.number_input("Día Ini:", 1, 31, 1, key="lab_dia_ini")
            with c_dia2: 
                dia_fin = st.number_input("Día Fin:", 1, 31, 15, key="lab_dia_fin")
        
        meses_atras_sim = st.slider("Meses atrás:", 2, 6, 3, key="lab_meses_atras")

        if st.button("🚀 Ejecutar", type="primary"):
            with st.spinner("Analizando..."):
                res = backtesting_estrategia_congelada(dfa, mes_sel_num, anio_sel, dia_ini, dia_fin, meses_atras_sim)
                
                if isinstance(res, dict):
                    st.success(f"✅ Efectividad: {res['% Efectividad Estructura']}%")
                    
                    col_izq, col_der = st.columns(2)
                    
                    with col_izq:
                        st.markdown("### 📋 Estrategia")
                        st.caption(f"Basada en: {res['Periodo Entrenamiento']}")
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🔥 Calientes D", str(res['hot_d']))
                        c2.metric("🟡 Tibios D", str(res['warm_d']))
                        c3.metric("🧊 Fríos D", str(res['cold_d']))
                        
                        c4, c5, c6 = st.columns(3)
                        c4.metric("🔥 Calientes U", str(res['hot_u']))
                        c5.metric("🟡 Tibios U", str(res['warm_u']))
                        c6.metric("🧊 Fríos U", str(res['cold_u']))
                        
                        st.markdown("**Perfiles Persistentes:**")
                        for p in res['Perfiles Persistentes']:
                            st.markdown(f"- 🏷️ {p}")

                    with col_der:
                        st.markdown("### 🎲 Resultados")
                        st.caption(f"Periodo: {res['Periodo Prueba']}")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total", res['Total Prueba'])
                        m2.metric("Aciertos", res['Aciertos Persistente'])
                        m3.metric("Sufrientes", res['Sufrientes Exitosos'])
                        
                        st.markdown("#### Detalle:")
                        df_view = res['Detalle'].copy()
                        df_view['Fecha'] = df_view['Fecha'].dt.strftime('%d/%m/%Y')
                        
                        def color_resultado(val):
                            if 'ESTRUCTURA' in str(val):
                                return 'background-color: #d4edda'
                            elif 'SUFRIENTE' in str(val):
                                return 'background-color: #fff3cd'
                            return 'background-color: #f8d7da'
                            
                        st.dataframe(
                            df_view.style.applymap(color_resultado, subset=['Resultado']),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.error(f"🛑 {res}")

    # PESTAÑA 5: ESTABILIDAD
    with tabs[5]:
        st.subheader(f"📉 Estabilidad: {t}")
        
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

if __name__ == "__main__":
    main()