import pandas as pd
import numpy as np
import math
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import requests
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def dia():
    from datetime import date, datetime, timedelta
    dia = date.today() + timedelta(0)
    return dia

def reset_index(df):
    df = df.reset_index(drop=True)
    df.index += 1
    return df

def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df
    
def grafico(df, nome):
    df = df.reset_index(drop=True)
    df.index += 1
    df['Profit_acu'] = df.Profit.cumsum()
    profit = round(df.Profit_acu.tail(1).item(),2)
    ROI = round((df.Profit_acu.tail(1)/len(df)*100).item(),2)
    df.Profit_acu.plot(title=nome, xlabel='Entradas', ylabel='Stakes')
    print("Profit:",profit,"stakes em", len(df),"jogos")
    print("ROI:",ROI,"%")

def money_finder(data,columns,target,divisions,min_jogos):
    def find_value(data,column,target,divisions):
        interval = (data[column].max() - data[column].min())/ divisions
        list_interval = list(np.arange(data[column].min(),data[column].max() + interval,interval))
        best_value = data[target].sum()
        best_data = data.copy()
        for x_min in list_interval[:-1]:
            for x_max in list_interval[list_interval.index(x_min)+1:]:
                some_data = data[data[column].between(x_min,x_max)]
                value = some_data[target].sum()
                if value > best_value:
                    best_value = value
                    best_data = some_data
        return (best_value,best_data,best_data.shape[0])
    best_value = data[target].sum()
    best_data = data.copy()
    df_parametros = pd.DataFrame()
    for column in columns:
        parameters = {}
        values = find_value(data = best_data,column=column,target=target,divisions=divisions)
        if (values[0] > best_value) & (values[2]>= min_jogos):
            best_value = values[0]
            best_data = values[1]
            for coluna in columns:
                parameters[f'{coluna.lower()}_min'] = best_data[coluna].min()
                parameters[f'{coluna.lower()}_max'] = best_data[coluna].max()
            parameters['n_games'] = values[2]
            parameters['value'] = best_value
            parameters = pd.DataFrame(parameters,index=[0])
            df_parametros = pd.concat([df_parametros,parameters],ignore_index=True)
            df_parametros = df_parametros.sort_values('value',ascending=False)
            df_parametros = df_parametros.iloc[:10]
    return df_parametros

def retorna_df(dicionario,data):
    data = data.copy()
    list_columns = dicionario.keys()
    for i in data.columns.to_list():
        for x in list_columns:
            if i.lower() in x:
                try:
                    x_min = dicionario[f'{i.lower()}_min']
                    x_max = dicionario[f'{i.lower()}_max']
                    data = data[data[i].between(x_min,x_max)]
                except:
                    pass
    return data
    
def calcula_prob(xg_casa,xg_visitante):
  prob_casa=[]
  prob_empate=[]
  prob_visitante=[]
  prob_btts=[]
  prob_o25=[]
  for gols_casa in range(0,15):
    for gols_visitante in range(0,15):
      prob = poisson.pmf(k=gols_casa,mu=xg_casa) * poisson.pmf(k=gols_visitante,mu=xg_visitante)
      if gols_casa > gols_visitante:
        prob_casa.append(prob)
      elif gols_casa == gols_visitante:
        prob_empate.append(prob)
      elif gols_casa < gols_visitante:
        prob_visitante.append(prob)
      else:
        pass
      if (gols_casa >0) & (gols_visitante >0):
        prob_btts.append(prob)
      else:
        pass
      if (gols_casa + gols_visitante) > 2.5:
        prob_o25.append(prob)
  return sum(prob_casa),sum(prob_empate),sum(prob_visitante),sum(prob_btts),sum(prob_o25)

def retorna_xg(dictionary):
    if (dictionary['FT_Odd_ML_H'] > 1) & (dictionary['FT_Odd_ML_D'] > 1) & (dictionary['FT_Odd_ML_A'] > 1) & (dictionary['FT_Odd_OU_O25'] > 1) & (dictionary['FT_Odd_BTTS_Yes'] > 1):
        odd_casa = dictionary['FT_Odd_ML_H']
        odd_empate = dictionary['FT_Odd_ML_D']
        odd_visitante = dictionary['FT_Odd_ML_A']
        odd_btts = dictionary['FT_Odd_BTTS_Yes']
        odd_o25 = dictionary['FT_Odd_OU_O25']
        def retorna_erro(xg_casa,xg_visitante,odd_casa=odd_casa,odd_empate=odd_empate,odd_visitante=odd_visitante,odd_btts=odd_btts,odd_o25=odd_o25):
            odd_casa = 1/odd_casa
            odd_empate = 1/odd_empate
            odd_visitante = 1/odd_visitante
            odd_btts = 1/odd_btts
            odd_o25 = 1/odd_o25
            p_casa,p_empate,p_visitante,p_btts,p_o25 = calcula_prob(xg_casa,xg_visitante)
            return -(abs(odd_casa - p_casa) + abs(odd_empate - p_empate) + abs(odd_visitante - p_visitante) + abs(odd_btts - p_btts) + abs(odd_o25 - p_o25))
        pbounds = {'xg_casa':(0,5),'xg_visitante':(0,5)}
        optimizer = BayesianOptimization(
            f=retorna_erro,
            pbounds=pbounds,
            verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=0,
            )
        otimizador = optimizer.maximize(
            init_points=10,
            n_iter= 200
            )
        dictionary['XG_Home'] = optimizer.max['params']['xg_casa']
        dictionary['XG_Away'] = optimizer.max['params']['xg_visitante']
        dictionary['XG_Erro'] = abs(optimizer.max['target'])