# python.exe -m pip install --upgrade pip
# pip install betfairlightweight bs4 fake_useragent openpyxl pandas pycaret schedule scikit-learn scipy selenium streamlit streamlit_authenticator telebot tinydb tqdm webdriver_manager

import betfairlightweight
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
import configparser
from io import BytesIO
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from pathlib import Path
import plotly.graph_objects as go
import pytz
from pycaret.classification import *
import requests
import schedule
import streamlit as st
import streamlit_authenticator as stauth
from scipy import stats
import telebot
import warnings
warnings.filterwarnings('ignore')

bot = telebot.TeleBot('6315876994:AAHWGy1p8y4yNvsAuo-xn12D-Guzncc6VNI')
chat_channel = '-1002100450066'

def ontem():
    from datetime import date, datetime, timedelta
    dia = date.today() - timedelta(1)
    return dia

def hoje():
    from datetime import date, datetime, timedelta
    dia = date.today() + timedelta(0)
    return dia

def amanha():
    from datetime import date, datetime, timedelta
    dia = date.today() + timedelta(1)
    return dia

def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df

def avg_total(df, team, date, var_for_column, var_against_column):
    prior_matches = df[(df['Home'] == team) | (df['Away'] == team)]
    prior_matches = prior_matches[prior_matches['Date'] < date].tail(5)

    var_for = 0
    var_against = 0

    for _, row in prior_matches.iterrows():
        if row['Home'] == team:
            var_for += row[var_for_column]
            var_against += row[var_against_column]
        else:
            var_for += row[var_against_column]
            var_against += row[var_for_column]

    num_games = len(prior_matches)
    avg_var_for = var_for / num_games if num_games > 0 else 0
    avg_var_against = var_against / num_games if num_games > 0 else 0

    return avg_var_for, avg_var_against

def plot_profit_acu(dataframe, title_text):
    dataframe['Profit_acu'] = dataframe.Profit.cumsum()
    n_apostas = dataframe.shape[0]
    profit = round(dataframe.Profit_acu.tail(1).item(), 2)
    ROI = round((dataframe.Profit_acu.tail(1) / n_apostas * 100).item(), 2)
    drawdown = dataframe['Profit_acu'] - dataframe['Profit_acu'].cummax()
    drawdown_maximo = round(drawdown.min(), 2)
    winrate_medio = round((dataframe['Profit'] > 0).mean() * 100, 2)
    desvio_padrao = round(dataframe['Profit'].std(), 2)
    dataframe.Profit_acu.plot(title=title_text, xlabel='Entradas', ylabel='Stakes')
    print("Metodo:",title_text)
    print("Profit:", profit, "stakes em", n_apostas, "jogos")
    print("ROI:", ROI, "%")
    print("Drawdown Maximo Acumulado:", drawdown_maximo)
    print("Winrate Medio:", winrate_medio, "%")
    print("Desvio Padrao:", desvio_padrao)
    print("")

def read_base(ligas):
    base = pd.read_csv("https://github.com/futpythontrader/YouTube/blob/main/Base_de_Dados/futpythontraderpunter.csv?raw=true")
    base["Date"] = pd.to_datetime(base["Date"])
    base = base[['League','Date','Home','Away','FT_Goals_H','FT_Goals_A','FT_Odd_H','FT_Odd_D','FT_Odd_A','FT_Odd_Over25','FT_Odd_Under25','Odd_BTTS_Yes','Odd_BTTS_No']]
    base.columns = ['League','Date','Home','Away','Goals_H','Goals_A','Odd_H','Odd_D','Odd_A','Odd_Over25','Odd_Under25','Odd_BTTS_Yes','Odd_BTTS_No']
    base = base[base['League'].isin(ligas) == True]
    base = drop_reset_index(base)
    return base

def read_jogos(dia, ligas):
    jogos_do_dia = pd.read_csv('https://github.com/futpythontrader/YouTube/blob/main/Jogos_do_Dia_FlashScore/'+str(dia)+'_Jogos_do_Dia_FlashScore.csv?raw=true')
    jogos_do_dia = jogos_do_dia[['League','Date','Time','Home','Away','FT_Odd_H','FT_Odd_D','FT_Odd_A','FT_Odd_Over25','FT_Odd_Under25','Odd_BTTS_Yes','Odd_BTTS_No']]
    jogos_do_dia.columns = ['League','Date','Time','Home','Away','Odd_H','Odd_D','Odd_A','Odd_Over25','Odd_Under25','Odd_BTTS_Yes','Odd_BTTS_No']
    jogos_do_dia = jogos_do_dia[jogos_do_dia['League'].isin(ligas) == True]
    jogos_do_dia = drop_reset_index(jogos_do_dia)
    return jogos_do_dia

def TR_Match_Odds(base, jogos_do_dia, lista):

    ligas = jogos_do_dia.sort_values(['League'])
    ligas = ligas['League'].unique()

    for i in ligas:
        liga = i

        df = base[base.League == liga]
        print(liga)

        df["Goals_H"] = df["Goals_H"].astype('int64')
        df["Goals_A"] = df["Goals_A"].astype('int64')
        df['Odd_H'] = pd.to_numeric(df['Odd_H'],errors = 'coerce')
        df['Odd_D'] = pd.to_numeric(df['Odd_D'],errors = 'coerce')
        df['Odd_A'] = pd.to_numeric(df['Odd_A'],errors = 'coerce')

        total_count = len(df.index)
    
        winHPLValues = 100 * df.Odd_H - 100
        winDPLValues = 100 * df.Odd_D - 100
        winAPLValues = 100 * df.Odd_A - 100
        losePLValues = -100

        df.loc[((df['Goals_H']) >  (df['Goals_A'])), 'Result'] = 'H'
        df.loc[((df['Goals_H']) == (df['Goals_A'])), 'Result'] = 'D'
        df.loc[((df['Goals_H'])  < (df['Goals_A'])), 'Result'] = 'A'

        df['H'] = winHPLValues.where(df.Result == 'H', other=losePLValues)
        df['D'] = winDPLValues.where(df.Result == 'D', other=losePLValues)
        df['A'] = winAPLValues.where(df.Result == 'A', other=losePLValues)

        no_of_days = 0

        matchDates = df.Date.unique()

        if no_of_days > 0:
            matchDates = (matchDates[-no_of_days:])

        df2 = pd.DataFrame()

        rowsIndex = []
        rowsDate = []
        rowsH = []
        rowsD = []
        rowsA = []

        count = 0
        for mDate in matchDates:
            count += 1
            rowsDate.append(mDate)
            rowsH.append(df.loc[df['Date'] == mDate]['H'].sum())
            rowsD.append(df.loc[df['Date'] == mDate]['D'].sum())
            rowsA.append(df.loc[df['Date'] == mDate]['A'].sum())

        df2['Date'] = rowsDate
        df2['H'] = rowsH
        df2['D'] = rowsD
        df2['A'] = rowsA

        df2 = df2.tail(101)
        df2 = df2.reset_index(drop=True)
        df2['Id'] = df2.reset_index()['index'].rename('index_copy')
        df2['Id'] = df2['Id'] + 1
        df2 = df2[['Id','Date','H','D','A']]

        df2['Hacu'] = df2.H.cumsum()
        df2['Dacu'] = df2.D.cumsum()
        df2['Aacu'] = df2.A.cumsum()

        df2['Hacu'].loc[0] = np.nan
        df2['Dacu'].loc[0] = np.nan
        df2['Aacu'].loc[0] = np.nan

        def weighted_mean_H(s):
            d = df2.loc[s.index, 'Hacu']
            w = df2.loc[s.index, 'Id']
            return (d * w).sum() / w.sum()

        def weighted_mean_H_C(s):
            d = df2.loc[s.index, 'waHC']
            w = df2.loc[s.index, 'Id']
            return (d * w).sum() / w.sum()

        def weighted_mean_D(s):
            d = df2.loc[s.index, 'Dacu']
            w = df2.loc[s.index, 'Id']
            return (d * w).sum() / w.sum()

        def weighted_mean_D_C(s):
            d = df2.loc[s.index, 'waDC']
            w = df2.loc[s.index, 'Id']
            return (d * w).sum() / w.sum()

        def weighted_mean_A(s):
            d = df2.loc[s.index, 'Aacu']
            w = df2.loc[s.index, 'Id']
            return (d * w).sum() / w.sum()

        def weighted_mean_A_C(s):
            d = df2.loc[s.index, 'waAC']
            w = df2.loc[s.index, 'Id']
            return (d * w).sum() / w.sum()

        df2['waH16'] = df2.rolling(16)['Hacu'].apply(weighted_mean_H, raw=False)
        df2['waH8'] = df2.rolling(8)['Hacu'].apply(weighted_mean_H, raw=False)
        df2['waHC'] = 2*df2.waH8-df2.waH16
        df2['waH4'] = df2.rolling(4)['waHC'].apply(weighted_mean_H_C, raw=False)

        df2['waD16'] = df2.rolling(16)['Dacu'].apply(weighted_mean_D, raw=False)
        df2['waD8'] = df2.rolling(8)['Dacu'].apply(weighted_mean_D, raw=False)
        df2['waDC'] = 2*df2.waD8-df2.waD16
        df2['waD4'] = df2.rolling(4)['waDC'].apply(weighted_mean_D_C, raw=False)

        df2['waA16'] = df2.rolling(16)['Aacu'].apply(weighted_mean_A, raw=False)
        df2['waA8'] = df2.rolling(8)['Aacu'].apply(weighted_mean_A, raw=False)
        df2['waAC'] = 2*df2.waA8-df2.waA16
        df2['waA4'] = df2.rolling(4)['waAC'].apply(weighted_mean_A_C, raw=False)

        df2['Hhull'] = df2['waH4']
        df2['Dhull'] = df2['waD4']
        df2['Ahull'] = df2['waA4']

        df2['Hdist'] = (df2.Hacu / df2.Hhull)
        df2['Ddist'] = (df2.Dacu / df2.Dhull)
        df2['Adist'] = (df2.Aacu / df2.Ahull)

        def r_function(s):
            f = s.loc[s.index.values[0]]
            l = s.loc[s.index.values[1]]
            return (l -f)/abs(f)

        df2['Hr'] = df2['Hhull'].rolling(2).apply(r_function, raw=False)
        df2['Dr'] = df2['Dhull'].rolling(2).apply(r_function, raw=False)
        df2['Ar'] = df2['Ahull'].rolling(2).apply(r_function, raw=False)

        def inc_H(s):
            x = df2.loc[s.index, 'Id']
            y = df2.loc[s.index, 'Hhull']
            slope, intercept = np.polyfit(x, y , 1)
            return slope

        def inc_D(s):
            x = df2.loc[s.index, 'Id']
            y = df2.loc[s.index, 'Dhull']
            slope, intercept = np.polyfit(x, y, 1)
            return slope

        def inc_A(s):
            x = df2.loc[s.index, 'Id']
            y = df2.loc[s.index, 'Ahull']
            slope, intercept = np.polyfit(x, y, 1)
            return slope

        df2['Hinc'] = df2['Hhull'].rolling(5).apply(inc_H, raw=False)
        df2['Dinc'] = df2['Dhull'].rolling(5).apply(inc_D, raw=False)
        df2['Ainc'] = df2['Ahull'].rolling(5).apply(inc_A, raw=False)

        df2['Hdp'] = df2['Hacu'].rolling(10).std()
        df2['Ddp'] = df2['Dacu'].rolling(10).std()
        df2['Adp'] = df2['Aacu'].rolling(10).std()

        df2['Hamp'] = df2['Hacu'].rolling(10).max() / df2['Hacu'].rolling(10).min()
        df2['Damp'] = df2['Dacu'].rolling(10).max() / df2['Dacu'].rolling(10).min()
        df2['Aamp'] = df2['Aacu'].rolling(10).max() / df2['Aacu'].rolling(10).min()

        df3 = pd.DataFrame()

        def normaliz(dfS):
            actual_value = (dfS.loc[dfS.index.values[4]])
            try:
                n = (actual_value - dfS.min()) / (dfS.max() - dfS.min())
                if math.isnan(n):
                    return 0
            except ZeroDivisionError:
                return 0
            return n

        df3['Id'] = df2['Id'].iloc[23:]

        df3['Hhull'] = df2['Hhull'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Dhull'] = df2['Dhull'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Ahull'] = df2['Ahull'].iloc[23:].rolling(5).apply(normaliz, raw=False)

        df3['Hdist'] = df2['Hdist'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Ddist'] = df2['Ddist'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Adist'] = df2['Adist'].iloc[23:].rolling(5).apply(normaliz, raw=False)

        df3['Hr'] = df2['Hr'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Dr'] = df2['Dr'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Ar'] = df2['Ar'].iloc[23:].rolling(5).apply(normaliz, raw=False)

        df3['Hinc'] = df2['Hinc'].rolling(5).apply(normaliz, raw=False)
        df3['Dinc'] = df2['Dinc'].rolling(5).apply(normaliz, raw=False)
        df3['Ainc'] = df2['Ainc'].rolling(5).apply(normaliz, raw=False)

        df3['Hdp'] = df2['Hdp'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Ddp'] = df2['Ddp'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Adp'] = df2['Adp'].iloc[23:].rolling(5).apply(normaliz, raw=False)

        df3['Hamp'] = df2['Hamp'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Damp'] = df2['Damp'].iloc[23:].rolling(5).apply(normaliz, raw=False)
        df3['Aamp'] = df2['Aamp'].iloc[23:].rolling(5).apply(normaliz, raw=False)

        df3['R'] = ''

        for index, row in df2.iterrows():
            if index > 26:
                try:
                    h = df2.iloc[(index+1)].H
                    d = df2.iloc[(index+1)].D
                    a = df2.iloc[(index+1)].A
                    if h > d and h > a:
                        df3.loc[index, 'R'] = 'H'
                    elif d > h and d > a:
                        df3.loc[index, 'R'] = 'D'
                    else:
                        df3.loc[index, 'R'] = 'A'
                except LookupError:
                    pass

        selected_date = df3.iloc[-1]

        distance_columns = ['Hhull', 'Dhull', 'Ahull', 'Hdist', 'Ddist', 'Adist', 'Hr', 'Dr', 'Ar', 'Hinc', 'Dinc', 'Ainc', 'Hdp',
                            'Ddp', 'Adp', 'Hamp', 'Damp', 'Aamp']

        def euclidean_distance(row):
            inner_value = 0
            for k in distance_columns:
                inner_value += (row[k] - selected_date[k]) ** 2
            return math.sqrt(inner_value)

        def euclidean_distance_tr(row):
            inner_value = 0
            for k in distance_columns:
                inner_value += abs(row[k] - selected_date[k])
            return inner_value

        df3['eucli'] = df3.apply(euclidean_distance_tr, axis=1)

        df4 = pd.DataFrame()
        df4['R'] = df3['R']
        df4['eucli'] = df3['eucli']
        df4.sort_values(by=['eucli'], inplace=True)

        teoria_dos_retornos = {}

        try:
            teoria_dos_retornos["League"] = liga
            teoria_dos_retornos["N1"] = df4[df4.eucli != 0]['R'].iloc[0]
        except:
            teoria_dos_retornos["N1"] = 0
        try:
            teoria_dos_retornos["N2"] = df4[df4.eucli != 0]['R'].iloc[1]
        except:
            teoria_dos_retornos["N2"] = 0
        try:
            teoria_dos_retornos["N3"] = df4[df4.eucli != 0]['R'].iloc[2]
        except:
            teoria_dos_retornos["N3"] = 0

        lista.append(teoria_dos_retornos)
    
ligas = ['ARGENTINA - LIGA PROFESIONAL',
'ARGENTINA - COPA DE LA LIGA PROFESIONAL',
'ARMENIA - PREMIER LEAGUE',
'AUSTRALIA - A-LEAGUE',
'AUSTRIA - 2. LIGA',
'AUSTRIA - BUNDESLIGA',
'BELGIUM - CHALLENGER PRO LEAGUE',
'BELGIUM - JUPILER PRO LEAGUE',
'BOSNIA AND HERZEGOVINA - PREMIJER LIGA BIH',
'BRAZIL - COPA DO BRASIL',
'BRAZIL - SERIE A',
'BRAZIL - SERIE B',
'BULGARIA - PARVA LIGA',
'CHINA - SUPER LEAGUE',
'CROATIA - HNL',
'CROATIA - PRVA NL',
'CZECH REPUBLIC - FORTUNA:LIGA',
'DENMARK - 1ST DIVISION',
'DENMARK - SUPERLIGA',
'EGYPT - PREMIER LEAGUE',
'ENGLAND - CHAMPIONSHIP',
'ENGLAND - LEAGUE ONE',
'ENGLAND - LEAGUE TWO',
'ENGLAND - NATIONAL LEAGUE',
'ENGLAND - PREMIER LEAGUE',
'ESTONIA - ESILIIGA',
'ESTONIA - MEISTRILIIGA',
'EUROPE - CHAMPIONS LEAGUE',
'EUROPE - EUROPA CONFERENCE LEAGUE',
'EUROPE - EUROPA LEAGUE',
'FINLAND - VEIKKAUSLIIGA',
'FINLAND - YKKONEN',
'FRANCE - LIGUE 1',
'FRANCE - LIGUE 2',
'FRANCE - NATIONAL',
'GERMANY - 2. BUNDESLIGA',
'GERMANY - 3. LIGA',
'GERMANY - BUNDESLIGA',
'HUNGARY - OTP BANK LIGA',
'ICELAND - BESTA DEILD KARLA',
'IRELAND - PREMIER DIVISION',
'ITALY - SERIE A',
'ITALY - SERIE B',
'JAPAN - J1 LEAGUE',
'JAPAN - J2 LEAGUE',
'MEXICO - LIGA DE EXPANSION MX',
'MEXICO - LIGA MX',
'NETHERLANDS - EERSTE DIVISIE',
'NETHERLANDS - EREDIVISIE',
'NORWAY - ELITESERIEN',
'NORWAY - OBOS-LIGAEN',
'POLAND - EKSTRAKLASA',
'PORTUGAL - LIGA PORTUGAL',
'PORTUGAL - LIGA PORTUGAL 2',
'ROMANIA - LIGA 1',
'SAUDI ARABIA - SAUDI PROFESSIONAL LEAGUE',
'SCOTLAND - CHAMPIONSHIP',
'SCOTLAND - LEAGUE ONE',
'SCOTLAND - LEAGUE TWO',
'SCOTLAND - PREMIERSHIP',
'SERBIA - SUPER LIGA',
'SLOVAKIA - NIKE LIGA',
'SLOVENIA - PRVA LIGA',
'SOUTH AMERICA - COPA LIBERTADORES',
'SOUTH AMERICA - COPA SUDAMERICANA',
'SOUTH KOREA - K LEAGUE 1',
'SOUTH KOREA - K LEAGUE 2',
'SPAIN - LALIGA',
'SPAIN - LALIGA2',
'SWEDEN - ALLSVENSKAN',
'SWEDEN - SUPERETTAN',
'SWITZERLAND - CHALLENGE LEAGUE',
'SWITZERLAND - SUPER LEAGUE',
'TURKEY - 1. LIG',
'TURKEY - SUPER LIG',
'UKRAINE - PREMIER LEAGUE',
'USA - MLS',
'WALES - CYMRU PREMIER']   

def rename_leagues(df):
    df.replace('Argentinian Copa de la Liga Profesional','ARGENTINA - COPA DE LA LIGA PROFESIONAL', inplace=True)
    df.replace('Argentinian Primera Division','ARGENTINA - LIGA PROFESIONAL', inplace=True)
    df.replace('Armenian Premier League','ARMENIA - PREMIER LEAGUE', inplace=True)
    df.replace('Australian A-League Men','AUSTRALIA - A-LEAGUE', inplace=True)
    df.replace('Austrian Bundesliga','AUSTRIA - BUNDESLIGA', inplace=True)
    df.replace('Austrian Erste Liga','AUSTRIA - 2. LIGA', inplace=True)
    df.replace('Belgian First Division A','BELGIUM - JUPILER PRO LEAGUE', inplace=True)
    df.replace('Belgian First Division B','BELGIUM - CHALLENGER PRO LEAGUE', inplace=True)
    df.replace('Bosnian Premier League','BOSNIA AND HERZEGOVINA - PREMIJER LIGA BIH', inplace=True)
    df.replace('Brazilian Serie A','BRAZIL - SERIE A', inplace=True)
    df.replace('Brazilian Serie B','BRAZIL - SERIE B', inplace=True)
    df.replace('Bulgarian A League','BULGARIA - PARVA LIGA', inplace=True)
    df.replace('Chinese League 1','CHINA - SUPER LEAGUE', inplace=True)
    df.replace('Croatian 1 HNL','CROATIA - HNL', inplace=True)
    df.replace('Croatian 2 HNL','CROATIA - PRVA NL', inplace=True)
    df.replace('Czech 1 Liga','CZECH REPUBLIC - FORTUNA:LIGA', inplace=True)
    df.replace('Danish 1st Division','DENMARK - 1ST DIVISION', inplace=True)
    df.replace('Danish Superliga','DENMARK - SUPERLIGA', inplace=True)
    df.replace('Dutch Eerste Divisie','NETHERLANDS - EERSTE DIVISIE', inplace=True)
    df.replace('Dutch Eredivisie','NETHERLANDS - EREDIVISIE', inplace=True)
    df.replace('Egyptian Premier','EGYPT - PREMIER LEAGUE', inplace=True)
    df.replace('English Championship','ENGLAND - CHAMPIONSHIP', inplace=True)
    df.replace('English League 1','ENGLAND - LEAGUE ONE', inplace=True)
    df.replace('English League 2','ENGLAND - LEAGUE TWO', inplace=True)
    df.replace('English National League','ENGLAND - NATIONAL LEAGUE', inplace=True)
    df.replace('English Premier League','ENGLAND - PREMIER LEAGUE', inplace=True)
    df.replace('Estonian Esiliiga','ESTONIA - ESILIIGA', inplace=True)
    df.replace('Estonian Premier League','ESTONIA - MEISTRILIIGA', inplace=True)
    df.replace('Finnish Veikkausliiga','FINLAND - VEIKKAUSLIIGA', inplace=True)
    df.replace('Finnish Ykkonen','FINLAND - YKKONEN', inplace=True)
    df.replace('French Ligue 1','FRANCE - LIGUE 1', inplace=True)
    df.replace('French Ligue 2','FRANCE - LIGUE 2', inplace=True)
    df.replace('French National','FRANCE - NATIONAL', inplace=True)
    df.replace('German 3 Liga','GERMANY - 3. LIGA', inplace=True)
    df.replace('German Bundesliga','GERMANY - BUNDESLIGA', inplace=True)
    df.replace('German Bundesliga 2','GERMANY - 2. BUNDESLIGA', inplace=True)
    df.replace('Hungarian NB I','HUNGARY - OTP BANK LIGA', inplace=True)
    df.replace('Icelandic Urvalsdeild','ICELAND - BESTA DEILD KARLA', inplace=True)
    df.replace('Irish Premier Division','IRELAND - PREMIER DIVISION', inplace=True)
    df.replace('Italian Serie A','ITALY - SERIE A', inplace=True)
    df.replace('Italian Serie B','ITALY - SERIE B', inplace=True)
    df.replace('Japanese J League','JAPAN - J1 LEAGUE', inplace=True)
    df.replace('Japanese J League 2','JAPAN - J2 LEAGUE', inplace=True)
    df.replace('Mexican Ascenso MX','MEXICO - LIGA DE EXPANSION MX', inplace=True)
    df.replace('Mexican Liga MX','MEXICO - LIGA MX', inplace=True)
    df.replace('Norwegian 1st Division','NORWAY - OBOS-LIGAEN', inplace=True)
    df.replace('Norwegian Eliteserien','NORWAY - ELITESERIEN', inplace=True)
    df.replace('Polish Ekstraklasa','POLAND - EKSTRAKLASA', inplace=True)
    df.replace('Portuguese Primeira Liga','PORTUGAL - LIGA PORTUGAL', inplace=True)
    df.replace('Portuguese Segunda Liga','PORTUGAL - LIGA PORTUGAL 2', inplace=True)
    df.replace('Romanian Liga I','ROMANIA - LIGA 1', inplace=True)
    df.replace('Saudi Professional League','SAUDI ARABIA - SAUDI PROFESSIONAL LEAGUE', inplace=True)
    df.replace('Scottish Championship','SCOTLAND - CHAMPIONSHIP', inplace=True)
    df.replace('Scottish League One','SCOTLAND - LEAGUE ONE', inplace=True)
    df.replace('Scottish League Two','SCOTLAND - LEAGUE TWO', inplace=True)
    df.replace('Scottish Premiership','SCOTLAND - PREMIERSHIP', inplace=True)
    df.replace('Serbian Super League','SERBIA - SUPER LIGA', inplace=True)
    df.replace('Slovakian Super League','SLOVAKIA - NIKE LIGA', inplace=True)
    df.replace('Slovenian Premier League','SLOVENIA - PRVA LIGA', inplace=True)
    df.replace('South Korean K League 1','SOUTH KOREA - K LEAGUE 1', inplace=True)
    df.replace('South Korean K League 2','SOUTH KOREA - K LEAGUE 2', inplace=True)
    df.replace('Spanish La Liga','SPAIN - LALIGA', inplace=True)
    df.replace('Spanish Segunda Division','SPAIN - LALIGA2', inplace=True)
    df.replace('Swedish Allsvenskan','SWEDEN - ALLSVENSKAN', inplace=True)
    df.replace('Swedish Superettan','SWEDEN - SUPERETTAN', inplace=True)
    df.replace('Swiss Challenge League','SWITZERLAND - CHALLENGE LEAGUE', inplace=True)
    df.replace('Swiss Super League','SWITZERLAND - SUPER LEAGUE', inplace=True)
    df.replace('Turkish 1 Lig','TURKEY - 1. LIG', inplace=True)
    df.replace('Turkish Super League','TURKEY - SUPER LIG', inplace=True)
    df.replace('Ukrainian Premier League','UKRAINE - PREMIER LEAGUE', inplace=True)
    df.replace('US Major League Soccer','USA - MLS', inplace=True)
    df.replace('Welsh Premiership','WALES - CYMRU PREMIER', inplace=True)
    
def rename_teams(df):
    df.replace('1860 Munich','Munich 1860', inplace=True)
    df.replace('AaB','Aalborg', inplace=True)
    df.replace('Aalesunds','Aalesund', inplace=True)
    df.replace('Aarau','Aarau', inplace=True)
    df.replace('ABC RN','ABC', inplace=True)
    df.replace('Aberdeen','Aberdeen', inplace=True)
    df.replace('Aberystwyth','Aberystwyth', inplace=True)
    df.replace('Abha','Abha', inplace=True)
    df.replace('AC Ajaccio','AC Ajaccio', inplace=True)
    df.replace('AC Horsens','Horsens', inplace=True)
    df.replace('AC Milan','AC Milan', inplace=True)
    df.replace('AC Monza','Monza', inplace=True)
    df.replace('AC Oulu','AC Oulu', inplace=True)
    df.replace('Academico de Viseu','Academico Viseu', inplace=True)
    df.replace('Accrington','Accrington', inplace=True)
    df.replace('ACS Petrolul 52','Petrolul', inplace=True)
    df.replace('ACS Sepsi OSK','Sepsi Sf. Gheorghe', inplace=True)
    df.replace('Adana Demirspor','Adana Demirspor', inplace=True)
    df.replace('Adanaspor','Adanaspor AS', inplace=True)
    df.replace('Adelaide United','Adelaide United', inplace=True)
    df.replace('Admira Wacker','Admira', inplace=True)
    df.replace('ADO Den Haag','Den Haag', inplace=True)
    df.replace('AFC Eskilstuna','AFC Eskilstuna', inplace=True)
    df.replace('AFC Fylde','AFC Fylde', inplace=True)
    df.replace('AFC Wimbledon','AFC Wimbledon', inplace=True)
    df.replace('AGF','Aarhus', inplace=True)
    df.replace('AIK','AIK', inplace=True)
    df.replace('Airdrieonians','Airdrieonians', inplace=True)
    df.replace('Ajax','Ajax', inplace=True)
    df.replace('Al Ahli','Al Ahli SC', inplace=True)
    df.replace('Al Ahly Cairo','Al Ahly', inplace=True)
    df.replace('Al Ittihad (EGY)','Al Ittihad', inplace=True)
    df.replace('Al Mokawloon','Arab Contractors', inplace=True)
    df.replace('Al Nasr FC Riyadh','Al Nassr', inplace=True)
    df.replace('Al Nassr','Al Nassr', inplace=True)
    df.replace('Al Riyadh SC','Al Riyadh', inplace=True)
    df.replace('Al Taee','Al Taee', inplace=True)
    df.replace('Al-Akhdoud','Al Akhdoud', inplace=True)
    df.replace('Alanyaspor','Alanyaspor', inplace=True)
    df.replace('Alashkert','Alashkert', inplace=True)
    df.replace('Alaves','Alaves', inplace=True)
    df.replace('Albacete','Albacete', inplace=True)
    df.replace('Albirex Niigata','Albirex Niigata', inplace=True)
    df.replace('Alcorcon','Alcorcon', inplace=True)
    df.replace('Aldershot','Aldershot', inplace=True)
    df.replace('Alebrijes De Oaxaca','Alebrijes Oaxaca', inplace=True)
    df.replace('Al-Ettifaq','Al Ettifaq', inplace=True)
    df.replace('Al-Fateh (KSA)','Al Fateh', inplace=True)
    df.replace('Al-Feiha','Al Feiha', inplace=True)
    df.replace('Al-Hazm (KSA)','Al Hazem', inplace=True)
    df.replace('Al-Hilal','Al Hilal', inplace=True)
    df.replace('Al-Ittihad','Al Ittihad', inplace=True)
    df.replace('Al-Khaleej Saihat','Al Khaleej', inplace=True)
    df.replace('Alloa','Alloa', inplace=True)
    df.replace('Al-Masry','Al Masry', inplace=True)
    df.replace('Almere City','Almere City', inplace=True)
    df.replace('Almeria','Almeria', inplace=True)
    df.replace('Al-Raed (KSA)','Al Raed', inplace=True)
    df.replace('Al-Shabab (KSA)','Al Shabab', inplace=True)
    df.replace('Al-Taawoun Buraidah','Al Taawon', inplace=True)
    df.replace('Altay','Altay', inplace=True)
    df.replace('Altrincham','Altrincham', inplace=True)
    df.replace('Al-Wahda (KSA)','Al Wehda', inplace=True)
    df.replace('America MG','America MG', inplace=True)
    df.replace('Amiens','Amiens', inplace=True)
    df.replace('Amorebieta','Amorebieta', inplace=True)
    df.replace('Anderlecht','Anderlecht', inplace=True)
    df.replace('Anderlecht B','Anderlecht U23', inplace=True)
    df.replace('Andorra CF','Andorra', inplace=True)
    df.replace('Angers','Angers', inplace=True)
    df.replace('Ankaragucu','Ankaragucu', inplace=True)
    df.replace('Annan','Annan', inplace=True)
    df.replace('Annecy','Annecy', inplace=True)
    df.replace('Ansan Greeners FC','Ansan Greeners', inplace=True)
    df.replace('Antalyaspor','Antalyaspor', inplace=True)
    df.replace('Antwerp','Antwerp', inplace=True)
    df.replace('Ararat Armenia','Ararat-Armenia', inplace=True)
    df.replace('Arbroath','Arbroath', inplace=True)
    df.replace('Arda','Arda', inplace=True)
    df.replace('Argentinos Juniors','Argentinos Jrs', inplace=True)
    df.replace('Arminia Bielefeld','Arminia Bielefeld', inplace=True)
    df.replace('Arouca','Arouca', inplace=True)
    df.replace('Arsenal','Arsenal', inplace=True)
    df.replace('Arsenal de Sarandi','Arsenal Sarandi', inplace=True)
    df.replace('Asane','Asane', inplace=True)
    df.replace('Ascoli','Ascoli', inplace=True)
    df.replace('Aston Villa','Aston Villa', inplace=True)
    df.replace('Atalanta','Atalanta', inplace=True)
    df.replace('Athletic Bilbao','Ath Bilbao', inplace=True)
    df.replace('Athletico-PR','Athletico-PR', inplace=True)
    df.replace('Atl Tucuman','Atl. Tucuman', inplace=True)
    df.replace('Atlanta Utd','Atlanta Utd', inplace=True)
    df.replace('Atlante','Atlante', inplace=True)
    df.replace('Atlas','Atlas', inplace=True)
    df.replace('Atletico GO','Atletico GO', inplace=True)
    df.replace('Atletico Madrid','Atl. Madrid', inplace=True)
    df.replace('Atletico MG','Atletico-MG', inplace=True)
    df.replace('Atletico San Luis','Atl. San Luis', inplace=True)
    df.replace('Augsburg','Augsburg', inplace=True)
    df.replace('Austin FC','Austin FC', inplace=True)
    df.replace('Austria Klagenfurt','A. Klagenfurt', inplace=True)
    df.replace('Austria Vienna','Austria Vienna', inplace=True)
    df.replace('Auxerre','Auxerre', inplace=True)
    df.replace('Avai','Avai', inplace=True)
    df.replace('Avranches','Avranches', inplace=True)
    df.replace('AVS Futebol SAD','AVS', inplace=True)
    df.replace('Ayr','Ayr', inplace=True)
    df.replace('AZ Alkmaar','AZ Alkmaar', inplace=True)
    df.replace('B93 Copenhagen','B.93', inplace=True)
    df.replace('Bahia','Bahia', inplace=True)
    df.replace('Bala Town','Bala', inplace=True)
    df.replace('Baladeyet Al-Mahalla','Baladiyat El Mahalla', inplace=True)
    df.replace('Bandirmaspor','Bandirmaspor', inplace=True)
    df.replace('Banfield','Banfield', inplace=True)
    df.replace('Banik Ostrava','Ostrava', inplace=True)
    df.replace('Barcelona','Barcelona', inplace=True)
    df.replace('Barnet','Barnet', inplace=True)
    df.replace('Barnsley','Barnsley', inplace=True)
    df.replace('Barracas Central','Barracas Central', inplace=True)
    df.replace('Barrow','Barrow', inplace=True)
    df.replace('Barry Town Utd','Barry', inplace=True)
    df.replace('Basaksehir','Basaksehir', inplace=True)
    df.replace('Bastia','Bastia', inplace=True)
    df.replace('Bayern Munich','Bayern Munich', inplace=True)
    df.replace('Beijing Guoan','Beijing Guoan', inplace=True)
    df.replace('Belediyesi Bodrumspor','Bodrumspor', inplace=True)
    df.replace('Belgrano','Belgrano', inplace=True)
    df.replace('Bellinzona','Bellinzona', inplace=True)
    df.replace('Benfica','Benfica', inplace=True)
    df.replace('Benfica B','Benfica B', inplace=True)
    df.replace('Beroe Stara Za','Beroe', inplace=True)
    df.replace('Besiktas','Besiktas', inplace=True)
    df.replace('Betis','Betis', inplace=True)
    df.replace('Birmingham','Birmingham', inplace=True)
    df.replace('BKMA Yerevan','BKMA', inplace=True)
    df.replace('Blackburn','Blackburn', inplace=True)
    df.replace('Blackpool','Blackpool', inplace=True)
    df.replace('Blaublitz Akita','Blaublitz', inplace=True)
    df.replace('Boavista','Boavista', inplace=True)
    df.replace('Boca Juniors','Boca Juniors', inplace=True)
    df.replace('Bochum','Bochum', inplace=True)
    df.replace('Bodo Glimt','Bodo/Glimt', inplace=True)
    df.replace('Bodrum Belediyesi Bodrumspor','Bodrumspor', inplace=True)
    df.replace('Bohemians','Bohemians', inplace=True)
    df.replace('Bohemians 1905','Bohemians', inplace=True)
    df.replace('Bologna','Bologna', inplace=True)
    df.replace('Bolton','Bolton', inplace=True)
    df.replace('Boluspor','Boluspor', inplace=True)
    df.replace('Bonnyrigg','Bonnyrigg Rose', inplace=True)
    df.replace('Borac Banja Luka','Borac Banja Luka', inplace=True)
    df.replace('Bordeaux','Bordeaux', inplace=True)
    df.replace('Boreham Wood','Boreham Wood', inplace=True)
    df.replace('Botafogo','Botafogo RJ', inplace=True)
    df.replace('Botafogo SP','Botafogo SP', inplace=True)
    df.replace('Botev Plovdiv','Botev Plovdiv', inplace=True)
    df.replace('Botev Vratsa','Botev Vratsa', inplace=True)
    df.replace('Botosani','FC Botosani', inplace=True)
    df.replace('Bournemouth','Bournemouth', inplace=True)
    df.replace('Bradford','Bradford City', inplace=True)
    df.replace('Braga','Braga', inplace=True)
    df.replace('Bragantino SP','Bragantino', inplace=True)
    df.replace('Brann','Brann', inplace=True)
    df.replace('Braunschweig','Braunschweig', inplace=True)
    df.replace('Bregenz','Bregenz', inplace=True)
    df.replace('Breidablik','Breidablik', inplace=True)
    df.replace('Brentford','Brentford', inplace=True)
    df.replace('Brescia','Brescia', inplace=True)
    df.replace('Brest','Brest', inplace=True)
    df.replace('Brighton','Brighton', inplace=True)
    df.replace('Brisbane Roar','Brisbane Roar', inplace=True)
    df.replace('Bristol City','Bristol City', inplace=True)
    df.replace('Bristol Rovers','Bristol Rovers', inplace=True)
    df.replace('Bromley','Bromley', inplace=True)
    df.replace('Brommapojkarna','Brommapojkarna', inplace=True)
    df.replace('Brondby','Brondby', inplace=True)
    df.replace('Bryne','Bryne', inplace=True)
    df.replace('BSK Bijelo Brdo','Bijelo Brdo', inplace=True)
    df.replace('Bucheon FC 1995','Bucheon FC 1995', inplace=True)
    df.replace('Burgos','Burgos CF', inplace=True)
    df.replace('Burnley','Burnley', inplace=True)
    df.replace('Burton Albion','Burton', inplace=True)
    df.replace('Busan IPark','Busan', inplace=True)
    df.replace('CA Independiente','Independiente', inplace=True)
    df.replace('CA Platense','Platense', inplace=True)
    df.replace('Cadiz','Cadiz CF', inplace=True)
    df.replace('Caen','Caen', inplace=True)
    df.replace('Caernarfon Town','Caernarfon', inplace=True)
    df.replace('Cagliari','Cagliari', inplace=True)
    df.replace('Cambridge Utd','Cambridge Utd', inplace=True)
    df.replace('Cambuur Leeuwarden','Cambuur', inplace=True)
    df.replace('Cancun FC','Cancun', inplace=True)
    df.replace('Cangzhou Mighty Lions','Cangzhou', inplace=True)
    df.replace('Cardiff','Cardiff', inplace=True)
    df.replace('Cardiff Metropolitan','Cardiff Metropolitan', inplace=True)
    df.replace('Carlisle','Carlisle', inplace=True)
    df.replace('Casa Pia','Casa Pia', inplace=True)
    df.replace('Catanzaro','Catanzaro', inplace=True)
    df.replace('CD Nacional Funchal','Nacional', inplace=True)
    df.replace('CD Tepatitlan De Morelos','Tepatitlan de Morelos', inplace=True)
    df.replace('Ceara SC Fortaleza','Ceara', inplace=True)
    df.replace('Celaya','Celaya', inplace=True)
    df.replace('Celta Vigo','Celta Vigo', inplace=True)
    df.replace('Celtic','Celtic', inplace=True)
    df.replace('Central Coast Mariners','Central Coast Mariners', inplace=True)
    df.replace('Central Cordoba (SdE)','Central Cordoba', inplace=True)
    df.replace('Ceramica Cleopatra','Ceramica Cleopatra', inplace=True)
    df.replace('Cercle Brugge','Cercle Brugge KSV', inplace=True)
    df.replace('Ceske Budejovice','Ceske Budejovice', inplace=True)
    df.replace('CF America','Club America', inplace=True)
    df.replace('CF Montreal','CF Montreal', inplace=True)
    df.replace('CF Os Belenenses','Os Belenenses', inplace=True)
    df.replace('CFR Cluj','CFR Cluj', inplace=True)
    df.replace('Changchun Yatai','Changchun Yatai', inplace=True)
    df.replace('Chapecoense','Chapecoense-SC', inplace=True)
    df.replace('Charleroi','Charleroi', inplace=True)
    df.replace('Charlotte FC','Charlotte', inplace=True)
    df.replace('Charlton','Charlton', inplace=True)
    df.replace('Chateauroux','Chateauroux', inplace=True)
    df.replace('Chaves','Chaves', inplace=True)
    df.replace('Chelsea','Chelsea', inplace=True)
    df.replace('Cheltenham','Cheltenham', inplace=True)
    df.replace('Chengdu Rongcheng','Chengdu Rongcheng', inplace=True)
    df.replace('Cheonan City','Cheonan City', inplace=True)
    df.replace('Cheongju FC','Cheongju', inplace=True)
    df.replace('Cherno More','Cherno More', inplace=True)
    df.replace('Chernomorets Odesa','Ch. Odessa', inplace=True)
    df.replace('Chesterfield','Chesterfield', inplace=True)
    df.replace('Chicago Fire','Chicago Fire', inplace=True)
    df.replace('Cholet SO','Cholet', inplace=True)
    df.replace('Chungnam Asan','Asan', inplace=True)
    df.replace('Cibalia Vinkovci','Cibalia', inplace=True)
    df.replace('Cimarrones de Sonora','Cimarrones de Sonora', inplace=True)
    df.replace('Cittadella','Cittadella', inplace=True)
    df.replace('Clermont','Clermont', inplace=True)
    df.replace('Club Atletico La Paz','Atletico La Paz', inplace=True)
    df.replace('Club Atletico Morelia','Atl. Morelia', inplace=True)
    df.replace('Club Brugge','Club Brugge KV', inplace=True)
    df.replace('Club Brugge B','Club Brugge KV U23', inplace=True)
    df.replace('Club Football Estrela','Estrela', inplace=True)
    df.replace('Clyde','Clyde', inplace=True)
    df.replace('Colchester','Colchester', inplace=True)
    df.replace('Colon','Colon Santa Fe', inplace=True)
    df.replace('Colorado','Colorado Rapids', inplace=True)
    df.replace('Columbus','Columbus Crew', inplace=True)
    df.replace('Colwyn Bay','Colwyn Bay', inplace=True)
    df.replace('Como','Como', inplace=True)
    df.replace('Concarneau','Concarneau', inplace=True)
    df.replace('Connahs Quay','Connahs Q.', inplace=True)
    df.replace('Corinthians','Corinthians', inplace=True)
    df.replace('Coritiba','Coritiba', inplace=True)
    df.replace('Cork City','Cork City', inplace=True)
    df.replace('Correcaminos UAT','Correcaminos', inplace=True)
    df.replace('Corum Belediyespor','Corum', inplace=True)
    df.replace('C-Osaka','Cerezo Osaka', inplace=True)
    df.replace('Cosenza','Cosenza', inplace=True)
    df.replace('Cove Rangers','Cove Rangers', inplace=True)
    df.replace('Coventry','Coventry', inplace=True)
    df.replace('Cracovia Krakow','Cracovia', inplace=True)
    df.replace('Crawley Town','Crawley', inplace=True)
    df.replace('CRB','CRB', inplace=True)
    df.replace('Crewe','Crewe', inplace=True)
    df.replace('Criciuma','Criciuma', inplace=True)
    df.replace('Cruz Azul','Cruz Azul', inplace=True)
    df.replace('Cruzeiro MG','Cruzeiro', inplace=True)
    df.replace('Crvena Zvezda','Crvena zvezda', inplace=True)
    df.replace('Crystal Palace','Crystal Palace', inplace=True)
    df.replace('CSKA 1948 Sofia','CSKA 1948 Sofia', inplace=True)
    df.replace('CSKA Sofia','CSKA Sofia', inplace=True)
    df.replace('CSMS Iasi','Poli Iasi', inplace=True)
    df.replace('Cuiaba','Cuiaba', inplace=True)
    df.replace('Cukaricki','Cukaricki', inplace=True)
    df.replace('Daegu FC','Daegu', inplace=True)
    df.replace('Daejeon Citizen','Daejeon', inplace=True)
    df.replace('Dag and Red','Dag & Red', inplace=True)
    df.replace('Dalian Yifang','Dalian Pro', inplace=True)
    df.replace('DC Utd','DC United', inplace=True)
    df.replace('De Graafschap','De Graafschap', inplace=True)
    df.replace('Debreceni VSC','Debrecen', inplace=True)
    df.replace('Defensa y Justicia','Defensa y Justicia', inplace=True)
    df.replace('Degerfors','Degerfors', inplace=True)
    df.replace('Deinze','Deinze', inplace=True)
    df.replace('Den Bosch','Den Bosch', inplace=True)
    df.replace('Derby','Derby', inplace=True)
    df.replace('Derry City','Derry City', inplace=True)
    df.replace('Dhamk','Damac', inplace=True)
    df.replace('Dijon','Dijon', inplace=True)
    df.replace('Dinamo Bucharest','Din. Bucuresti', inplace=True)
    df.replace('Dinamo Zagreb','D. Zagreb', inplace=True)
    df.replace('Diosgyori','DVTK', inplace=True)
    df.replace('Djurgardens','Djurgarden', inplace=True)
    df.replace('Dnipro-1','Dnipro-1', inplace=True)
    df.replace('Domzale','Domzale', inplace=True)
    df.replace('Doncaster','Doncaster', inplace=True)
    df.replace('Dorados','Dorados de Sinaloa', inplace=True)
    df.replace('Dorking Wanderers','Dorking', inplace=True)
    df.replace('Dortmund','Dortmund', inplace=True)
    df.replace('Dortmund II','Dortmund II', inplace=True)
    df.replace('Drogheda','Drogheda', inplace=True)
    df.replace('DSV Leoben','Leoben', inplace=True)
    df.replace('Duisburg','Duisburg', inplace=True)
    df.replace('Dukla Banska Bystrica','Banska Bystrica', inplace=True)
    df.replace('Dumbarton','Dumbarton', inplace=True)
    df.replace('Dunajska Streda','Dun. Streda', inplace=True)
    df.replace('Dundalk','Dundalk', inplace=True)
    df.replace('Dundee','Dundee FC', inplace=True)
    df.replace('Dundee Utd','Dundee FC', inplace=True)
    df.replace('Dundee Utd','Dundee Utd', inplace=True)
    df.replace('Dunfermline','Dunfermline', inplace=True)
    df.replace('Dunkerque','Dunkerque', inplace=True)
    df.replace('Dynamo Dresden','SG Dynamo Dresden', inplace=True)
    df.replace('Dynamo Kiev','Dyn. Kyiv', inplace=True)
    df.replace('East Fife','East Fife', inplace=True)
    df.replace('Eastleigh','Eastleigh', inplace=True)
    df.replace('Ebbsfleet Utd','Ebbsfleet', inplace=True)
    df.replace('EC Vitoria Salvador','Vitoria', inplace=True)
    df.replace('Edinburgh City','Edinburgh City', inplace=True)
    df.replace('Eibar','Eibar', inplace=True)
    df.replace('EIF','Ekenas', inplace=True)
    df.replace('Eintracht Frankfurt','Eintracht Frankfurt', inplace=True)
    df.replace('El Daklyeh','El Daklyeh', inplace=True)
    df.replace('El Geish','El Gaish', inplace=True)
    df.replace('El Gounah','El Gouna', inplace=True)
    df.replace('Elche','Elche', inplace=True)
    df.replace('Eldense','Eldense', inplace=True)
    df.replace('Elfsborg','Elfsborg', inplace=True)
    df.replace('Elgin City FC','Elgin City', inplace=True)
    df.replace('Elversberg','Elversberg', inplace=True)
    df.replace('Emmen','FC Emmen', inplace=True)
    df.replace('Empoli','Empoli', inplace=True)
    df.replace('ENPPI','Enppi', inplace=True)
    df.replace('Epinal','Epinal', inplace=True)
    df.replace('Erzgebirge','Aue', inplace=True)
    df.replace('Erzurum BB','Erzurumspor', inplace=True)
    df.replace('Espanyol','Espanyol', inplace=True)
    df.replace('ESTAC Troyes','Troyes', inplace=True)
    df.replace('Estoril Praia','Estoril', inplace=True)
    df.replace('Estudiantes','Estudiantes L.P.', inplace=True)
    df.replace('Etar','Etar', inplace=True)
    df.replace('Eupen','Eupen', inplace=True)
    df.replace('Everton','Everton', inplace=True)
    df.replace('Excelsior','Excelsior', inplace=True)
    df.replace('Exeter','Exeter', inplace=True)
    df.replace('Eyupspor','Eyupspor', inplace=True)
    df.replace('Falkirk','Falkirk', inplace=True)
    df.replace('Famalicao','Famalicao', inplace=True)
    df.replace('Farense','SC Farense', inplace=True)
    df.replace('Farul Constanta','Farul Constanta', inplace=True)
    df.replace('Fatih Karagumruk Istanbul','Karagumruk', inplace=True)
    df.replace('FC Anyang','Anyang', inplace=True)
    df.replace('FC Ararat Yerevan','Ararat Yerevan', inplace=True)
    df.replace('Fc Baden','Baden', inplace=True)
    df.replace('FC Basel','Basel', inplace=True)
    df.replace('FC Blau Weiss Linz','BW Linz', inplace=True)
    df.replace('FC Cartagena','FC Cartagena SAD', inplace=True)
    df.replace('FC Cincinnati','FC Cincinnati', inplace=True)
    df.replace('FC Copenhagen','FC Copenhagen', inplace=True)
    df.replace('FC Dallas','FC Dallas', inplace=True)
    df.replace('FC Dordrecht','Dordrecht', inplace=True)
    df.replace('Fc Dornbirn','Dornbirn', inplace=True)
    df.replace('FC Eindhoven','Eindhoven FC', inplace=True)
    df.replace('FC Elva','Elva', inplace=True)
    df.replace('FC Groningen','Groningen', inplace=True)
    df.replace('FC Halifax Town','FC Halifax', inplace=True)
    df.replace('FC Heidenheim','Heidenheim', inplace=True)
    df.replace('FC Helsingor','Helsingor', inplace=True)
    df.replace('FC Inter','Inter Turku', inplace=True)
    df.replace('FC Juarez','Juarez', inplace=True)
    df.replace('FC Koln','FC Koln', inplace=True)
    df.replace('FC Kosice','Kosice', inplace=True)
    df.replace('FC Liefering','Liefering', inplace=True)
    df.replace('FC Liege','RFC Liege', inplace=True)
    df.replace('FC Machida','Machida', inplace=True)
    df.replace('FC Magdeburg','Magdeburg', inplace=True)
    df.replace('FC Minaj','Minaj', inplace=True)
    df.replace('FC Noah','Noah', inplace=True)
    df.replace('FC Nordsjaelland','Nordsjaelland', inplace=True)
    df.replace('FC Oss','Oss', inplace=True)
    df.replace('FC Pyunik','Pyunik Yerevan', inplace=True)
    df.replace('FC Seoul','Seoul', inplace=True)
    df.replace('FC Shirak','Shirak Gyumri', inplace=True)
    df.replace('FC Tallinn','FC Tallinn', inplace=True)
    df.replace('FC Tokyo','FC Tokyo', inplace=True)
    df.replace('FC Twente','Twente', inplace=True)
    df.replace('FC U Craiova 1948','U Craiova 1948', inplace=True)
    df.replace('FC Urartu','Urartu', inplace=True)
    df.replace('FC Utrecht','Utrecht', inplace=True)
    df.replace('FC Vaduz','Vaduz', inplace=True)
    df.replace('FC Van Yerevan','Van', inplace=True)
    df.replace('FC Volendam','FC Volendam', inplace=True)
    df.replace('FC Voluntari','FC Voluntari', inplace=True)
    df.replace('FC Wil','Wil', inplace=True)
    df.replace('FC Zurich','Zurich', inplace=True)
    df.replace('FCI Tallinn','Levadia', inplace=True)
    df.replace('FCSB','FCSB', inplace=True)
    df.replace('FCV Dender','Dender', inplace=True)
    df.replace('Feirense','Feirense', inplace=True)
    df.replace('Fenerbahce','Fenerbahce', inplace=True)
    df.replace('Feralpisalo','FeralpiSalo', inplace=True)
    df.replace('Ferencvaros','Ferencvaros', inplace=True)
    df.replace('Feyenoord','Feyenoord', inplace=True)
    df.replace('Fiorentina','Fiorentina', inplace=True)
    df.replace('First Vienna Fc 1894','First Vienna', inplace=True)
    df.replace('FK Backa Topola','TSC', inplace=True)
    df.replace('FK Igman Konjic','Igman K.', inplace=True)
    df.replace('FK IMT Novi Beograd','IMT Novi Beograd', inplace=True)
    df.replace('FK Jablonec','Jablonec', inplace=True)
    df.replace('FK Javor Ivanjica','Javor', inplace=True)
    df.replace('FK Napredak','Napredak', inplace=True)
    df.replace('FK Novi Pazar','Novi Pazar', inplace=True)
    df.replace('FK Radnicki 1923','Radnicki 1923', inplace=True)
    df.replace('FK Spartak','Sp. Subotica', inplace=True)
    df.replace('FK Velez Mostar','Velez Mostar', inplace=True)
    df.replace('Flamengo','Flamengo RJ', inplace=True)
    df.replace('Fleetwood Town','Fleetwood', inplace=True)
    df.replace('Flora Tallinn II','Flora U21', inplace=True)
    df.replace('Floridsdorfer AC','Floridsdorfer AC', inplace=True)
    df.replace('Fluminense','Fluminense', inplace=True)
    df.replace('Forest Green','Forest Green', inplace=True)
    df.replace('Forfar','Forfar Athletic', inplace=True)
    df.replace('Fortaleza EC','Fortaleza', inplace=True)
    df.replace('Fortuna Dusseldorf','Dusseldorf', inplace=True)
    df.replace('Fortuna Sittard','Sittard', inplace=True)
    df.replace('Fram','Fram', inplace=True)
    df.replace('Francs Borains','Francs Borains', inplace=True)
    df.replace('Fredericia','Fredericia', inplace=True)
    df.replace('Fredrikstad','Fredrikstad', inplace=True)
    df.replace('Freiburg','Freiburg', inplace=True)
    df.replace('Freiburg II','Freiburg II', inplace=True)
    df.replace('Frosinone','Frosinone', inplace=True)
    df.replace('Fujieda Myfc','Fujieda MYFC', inplace=True)
    df.replace('Fukuoka','Avispa Fukuoka', inplace=True)
    df.replace('Fulham','Fulham', inplace=True)
    df.replace('Future FC','Future FC', inplace=True)
    df.replace('Fylkir','Fylkir', inplace=True)
    df.replace('GAIS','GAIS', inplace=True)
    df.replace('Galatasaray','Galatasaray', inplace=True)
    df.replace('Gangwon','Gangwon', inplace=True)
    df.replace('Gateshead','Gateshead', inplace=True)
    df.replace('Gaziantep FK','Gaziantep', inplace=True)
    df.replace('Gefle','Gefle', inplace=True)
    df.replace('Genclerbirligi','Genclerbirligi', inplace=True)
    df.replace('Genk','Genk', inplace=True)
    df.replace('Genoa','Genoa', inplace=True)
    df.replace('Gent','Gent', inplace=True)
    df.replace('Getafe','Getafe', inplace=True)
    df.replace('Gil Vicente','Gil Vicente', inplace=True)
    df.replace('Gillingham','Gillingham', inplace=True)
    df.replace('Gimcheon Sangmu','Gimcheon Sangmu', inplace=True)
    df.replace('Gimnasia La Plata','Gimnasia L.P.', inplace=True)
    df.replace('Gimpo Citizen','Gimpo FC', inplace=True)
    df.replace('Giresunspor','Giresunspor', inplace=True)
    df.replace('Girona','Girona', inplace=True)
    df.replace('Gnistan','Gnistan', inplace=True)
    df.replace('Gnistan','Gnistan', inplace=True)
    df.replace('Go Ahead Eagles','G.A. Eagles', inplace=True)
    df.replace('GOAL FC','GOAL FC', inplace=True)
    df.replace('Godoy Cruz','Godoy Cruz', inplace=True)
    df.replace('Goias','Goias', inplace=True)
    df.replace('Gornik Zabrze','Gornik Zabrze', inplace=True)
    df.replace('G-Osaka','Gamba Osaka', inplace=True)
    df.replace('GOSK Gabela','GOSK Gabela', inplace=True)
    df.replace('Goztepe','Goztepe', inplace=True)
    df.replace('Granada','Granada CF', inplace=True)
    df.replace('Grasshoppers Zurich','Grasshoppers', inplace=True)
    df.replace('Grazer AK','Grazer AK', inplace=True)
    df.replace('Gremio','Gremio', inplace=True)
    df.replace('Grenoble','Grenoble', inplace=True)
    df.replace('Greuther Furth','Greuther Furth', inplace=True)
    df.replace('Grimsby','Grimsby', inplace=True)
    df.replace('Guadalajara','Guadalajara Chivas', inplace=True)
    df.replace('Guarani','Guarani', inplace=True)
    df.replace('Guimaraes','Vitoria Guimaraes', inplace=True)
    df.replace('Guingamp','Guingamp', inplace=True)
    df.replace('Gwangju FC','Gwangju FC', inplace=True)
    df.replace('Gyeongnam','Gyeongnam', inplace=True)
    df.replace('Hacken','Hacken', inplace=True)
    df.replace('Hafnarfjordur','Hafnarfjordur', inplace=True)
    df.replace('Hajduk Split','Hajduk Split', inplace=True)
    df.replace('Haka','Haka', inplace=True)
    df.replace('Hallescher FC','Hallescher', inplace=True)
    df.replace('Halmstads','Halmstad', inplace=True)
    df.replace('Hamburger SV','Hamburger SV', inplace=True)
    df.replace('Hamilton','Hamilton', inplace=True)
    df.replace('Ham-Kam','HamKam', inplace=True)
    df.replace('Hammarby','Hammarby', inplace=True)
    df.replace('Hannover','Hannover', inplace=True)
    df.replace('Hansa Rostock','Hansa Rostock', inplace=True)
    df.replace('Harju JK Laagri','Harju JK Laagri', inplace=True)
    df.replace('Harrogate Town','Harrogate', inplace=True)
    df.replace('Hartberg','Hartberg', inplace=True)
    df.replace('Hartlepool','Hartlepool', inplace=True)
    df.replace('Hatayspor','Hatayspor', inplace=True)
    df.replace('Haugesund','Haugesund', inplace=True)
    df.replace('Haverfordwest County','Haverfordwest', inplace=True)
    df.replace('HB Koge','Koge', inplace=True)
    df.replace('Hearts','Hearts', inplace=True)
    df.replace('Hebar','Hebar', inplace=True)
    df.replace('Heerenveen','Heerenveen', inplace=True)
    df.replace('Helmond Sport','Helmond', inplace=True)
    df.replace('Helsingborgs','Helsingborg', inplace=True)
    df.replace('Henan Songshan Longmen','Henan Songshan Longmen', inplace=True)
    df.replace('Heracles','Heracles', inplace=True)
    df.replace('Hermannstadt','FC Hermannstadt', inplace=True)
    df.replace('Hertha Berlin','Hertha Berlin', inplace=True)
    df.replace('Hibernian','Hibernian', inplace=True)
    df.replace('HIFK','HIFK', inplace=True)
    df.replace('Hillerod Fodbold','Hillerod', inplace=True)
    df.replace('Hiroshima','Sanfrecce Hiroshima', inplace=True)
    df.replace('HJK Helsinki','HJK', inplace=True)
    df.replace('HK Kopavogur','Kopavogur', inplace=True)
    df.replace('HNK Gorica','Gorica', inplace=True)
    df.replace('HNK Orijent 1919','Orijent', inplace=True)
    df.replace('Hobro','Hobro', inplace=True)
    df.replace('Hodd','Hodd', inplace=True)
    df.replace('Hoffenheim','Hoffenheim', inplace=True)
    df.replace('Holstein Kiel','Holstein Kiel', inplace=True)
    df.replace('Honka','Honka', inplace=True)
    df.replace('Houston Dynamo','Houston Dynamo', inplace=True)
    df.replace('Hradec Kralove','Hradec Kralove', inplace=True)
    df.replace('Huddersfield','Huddersfield', inplace=True)
    df.replace('Huesca','Huesca', inplace=True)
    df.replace('Hull','Hull', inplace=True)
    df.replace('Huracan','Huracan', inplace=True)
    df.replace('Hvidovre','Hvidovre IF', inplace=True)
    df.replace('IBV','Vestmannaeyjar', inplace=True)
    df.replace('IFK Goteborg','Goteborg', inplace=True)
    df.replace('IFK Mariehamn','Mariehamn', inplace=True)
    df.replace('IK Brage','Brage', inplace=True)
    df.replace('Ilves','Ilves', inplace=True)
    df.replace('Incheon Utd','Incheon', inplace=True)
    df.replace('Ingolstadt','Ingolstadt', inplace=True)
    df.replace('Instituto','Instituto', inplace=True)
    df.replace('Inter','Inter', inplace=True)
    df.replace('Inter Miami CF','Inter Miami', inplace=True)
    df.replace('Internacional','Internacional', inplace=True)
    df.replace('Inverness CT','Inverness', inplace=True)
    df.replace('Ipswich','Ipswich', inplace=True)
    df.replace('Ismaily','El Ismaily', inplace=True)
    df.replace('Istanbulspor','Istanbulspor AS', inplace=True)
    df.replace('Ituano','Ituano', inplace=True)
    df.replace('Iwaki SC','Iwaki', inplace=True)
    df.replace('Iwata','Iwata', inplace=True)
    df.replace('Jagiellonia Bialystock','Jagiellonia', inplace=True)
    df.replace('Jahn Regensburg','Regensburg', inplace=True)
    df.replace('JaPS','JaPS', inplace=True)
    df.replace('Jaro','Jaro', inplace=True)
    df.replace('Jef Utd Chiba','Chiba', inplace=True)
    df.replace('Jeju Utd','Jeju Utd', inplace=True)
    df.replace('Jeonbuk Motors','Jeonbuk', inplace=True)
    df.replace('Jeonnam Dragons','Jeonnam', inplace=True)
    df.replace('Jerv','Jerv', inplace=True)
    df.replace('JJK','JJK Jyvaskyla', inplace=True)
    df.replace('Jong Ajax Amsterdam','Jong Ajax', inplace=True)
    df.replace('Jong AZ Alkmaar','Jong AZ', inplace=True)
    df.replace('Jong FC Utrecht','Jong Utrecht', inplace=True)
    df.replace('Jong PSV Eindhoven','Jong PSV', inplace=True)
    df.replace('Jonkopings Sodra','Jonkoping', inplace=True)
    df.replace('Juventude','Juventude', inplace=True)
    df.replace('Juventus','Juventus', inplace=True)
    df.replace('KA Akureyri','KA Akureyri', inplace=True)
    df.replace('Kaiserslautern','Kaiserslautern', inplace=True)
    df.replace('Kalmar FF','Kalmar', inplace=True)
    df.replace('Kanazawa','Kanazawa', inplace=True)
    df.replace('Kansas City','Sporting Kansas City', inplace=True)
    df.replace('KaPa','KaPa', inplace=True)
    df.replace('Karlsruhe','Karlsruher SC', inplace=True)
    df.replace('Kashima','Kashima Antlers', inplace=True)
    df.replace('Kashiwa','Kashiwa Reysol', inplace=True)
    df.replace('Kasimpasa','Kasimpasa', inplace=True)
    df.replace('Kawasaki','Kawasaki Frontale', inplace=True)
    df.replace('Kayserispor','Kayserispor', inplace=True)
    df.replace('Keciorengucu','Keciorengucu', inplace=True)
    df.replace('Kecskemeti','Kecskemeti TE', inplace=True)
    df.replace('Keflavik','Keflavik', inplace=True)
    df.replace('Kelty Hearts','Kelty Hearts', inplace=True)
    df.replace('KFCO Beerschot Wilrijk','Beerschot VA', inplace=True)
    df.replace('KFUM Oslo','KFUM Oslo', inplace=True)
    df.replace('Kidderminster','Kidderminster', inplace=True)
    df.replace('Kilmarnock','Kilmarnock', inplace=True)
    df.replace('Kisvarda','Kisvarda', inplace=True)
    df.replace('Kobe','Vissel Kobe', inplace=True)
    df.replace('Kocaelispor','Kocaelispor', inplace=True)
    df.replace('Kofu','Kofu', inplace=True)
    df.replace('Kohtla-Jarve','FC Alliance', inplace=True)
    df.replace('Kolding IF','Kolding IF', inplace=True)
    df.replace('Kolos Kovalyovka','Kolos Kovalivka', inplace=True)
    df.replace('Kongsvinger','Kongsvinger', inplace=True)
    df.replace('Konyaspor','Konyaspor', inplace=True)
    df.replace('Koper','Koper', inplace=True)
    df.replace('Korona Kielce','Korona Kielce', inplace=True)
    df.replace('Kortrijk','Kortrijk', inplace=True)
    df.replace('KPV','KPV Kokkola', inplace=True)
    df.replace('KR Reykjavik','KR Reykjavik', inplace=True)
    df.replace('Kristiansund','Kristiansund', inplace=True)
    df.replace('Kryvbas Krivyi Rih','Kryvbas', inplace=True)
    df.replace('KSV 1919','Kapfenberg', inplace=True)
    df.replace('KTP','KTP', inplace=True)
    df.replace('Kumamoto','Kumamoto', inplace=True)
    df.replace('KuPS','KuPS', inplace=True)
    df.replace('Kuressaare','Kuressaare', inplace=True)
    df.replace('KV Oostende','Oostende', inplace=True)
    df.replace('Kyoto','Kyoto', inplace=True)
    df.replace('LA Galaxy','Los Angeles Galaxy', inplace=True)
    df.replace('Lahti','Lahti', inplace=True)
    df.replace('Landskrona','Landskrona', inplace=True)
    df.replace('Lanus','Lanus', inplace=True)
    df.replace('Las Palmas','Las Palmas', inplace=True)
    df.replace('LASK Linz','LASK', inplace=True)
    df.replace('Lausanne','Lausanne', inplace=True)
    df.replace('Laval','Laval', inplace=True)
    df.replace('Lazio','Lazio', inplace=True)
    df.replace('Le Havre','Le Havre', inplace=True)
    df.replace('Le Mans','Le Mans', inplace=True)
    df.replace('Lecce','Lecce', inplace=True)
    df.replace('Lecco','Lecco', inplace=True)
    df.replace('Lech Poznan','Lech Poznan', inplace=True)
    df.replace('Leeds','Leeds', inplace=True)
    df.replace('Leganes','Leganes', inplace=True)
    df.replace('Legia Warsaw','Legia', inplace=True)
    df.replace('Leicester','Leicester', inplace=True)
    df.replace('Leiria','Leiria', inplace=True)
    df.replace('Leixoes','Leixoes', inplace=True)
    df.replace('Lens','Lens', inplace=True)
    df.replace('Leon','Club Leon', inplace=True)
    df.replace('Leones Negros','Leones Negros', inplace=True)
    df.replace('Levadia Tallinn II','Levadia U21', inplace=True)
    df.replace('Levante','Levante', inplace=True)
    df.replace('Leverkusen','Bayer Leverkusen', inplace=True)
    df.replace('Levski Krumovgrad','Krumovgrad', inplace=True)
    df.replace('Leyton Orient','Leyton Orient', inplace=True)
    df.replace('Lierse','Lierse K.', inplace=True)
    df.replace('Lille','Lille', inplace=True)
    df.replace('Lillestrom','Lillestrom', inplace=True)
    df.replace('Lincoln','Lincoln', inplace=True)
    df.replace('Liverpool','Liverpool', inplace=True)
    df.replace('Livingston','Livingston', inplace=True)
    df.replace('LKS Lodz','LKS Lodz', inplace=True)
    df.replace('LNZ-Lebedyn','LNZ Cherkasy', inplace=True)
    df.replace('Lokomotiv Plovdiv','Lok. Plovdiv', inplace=True)
    df.replace('Lokomotiv Sofia','Lok. Sofia', inplace=True)
    df.replace('Lokomotiva','Lok. Zagreb', inplace=True)
    df.replace('Lommel','Lommel SK', inplace=True)
    df.replace('Londrina','Londrina', inplace=True)
    df.replace('Lorient','Lorient', inplace=True)
    df.replace('Los Angeles FC','Los Angeles FC', inplace=True)
    df.replace('Lubeck','Lubeck', inplace=True)
    df.replace('Ludogorets','Ludogorets', inplace=True)
    df.replace('Lugano','Lugano', inplace=True)
    df.replace('Luton','Luton', inplace=True)
    df.replace('Luzern','Luzern', inplace=True)
    df.replace('Lyngby','Lyngby', inplace=True)
    df.replace('Lyon','Lyon', inplace=True)
    df.replace('Macarthur FC','Macarthur FC', inplace=True)
    df.replace('Mafra','Mafra', inplace=True)
    df.replace('Maidenhead','Maidenhead', inplace=True)
    df.replace('Mainz','Mainz', inplace=True)
    df.replace('Mallorca','Mallorca', inplace=True)
    df.replace('Malmo FF','Malmo FF', inplace=True)
    df.replace('Man City','Manchester City', inplace=True)
    df.replace('Man Utd','Manchester Utd', inplace=True)
    df.replace('Manisa FK','Manisa FK', inplace=True)
    df.replace('Mansfield','Mansfield', inplace=True)
    df.replace('Marignane-Gignac','Marignane', inplace=True)
    df.replace('Maritimo','Maritimo', inplace=True)
    df.replace('Marseille','Marseille', inplace=True)
    df.replace('Martigues','Martigues', inplace=True)
    df.replace('Mazatlan FC','Mazatlan FC', inplace=True)
    df.replace('Meizhou Hakka','Meizhou Hakka', inplace=True)
    df.replace('Melbourne City','Melbourne City', inplace=True)
    df.replace('Melbourne Victory','Melbourne Victory', inplace=True)
    df.replace('Metalist 1925','Metalist 1925', inplace=True)
    df.replace('Metz','Metz', inplace=True)
    df.replace('Mezokovesd-Zsory','Mezokovesd-Zsory', inplace=True)
    df.replace('MFK Karvina','Karvina', inplace=True)
    df.replace('MFK Kosice','Kosice', inplace=True)
    df.replace('MFK Skalica','Skalica', inplace=True)
    df.replace('Mgladbach','B. Monchengladbach', inplace=True)
    df.replace('Middlesbrough','Middlesbrough', inplace=True)
    df.replace('Midtjylland','Midtjylland', inplace=True)
    df.replace('Mikkeli','Mikkeli', inplace=True)
    df.replace('Mikkeli','Mikkeli', inplace=True)
    df.replace('Millwall','Millwall', inplace=True)
    df.replace('Mineros de Zacatecas','Zacatecas Mineros', inplace=True)
    df.replace('Minnesota Utd','Minnesota United', inplace=True)
    df.replace('Mirandes','Mirandes', inplace=True)
    df.replace('Mirassol','Mirassol', inplace=True)
    df.replace('Mito','Mito', inplace=True)
    df.replace('Mjallby','Mjallby', inplace=True)
    df.replace('Mjondalen','Mjondalen', inplace=True)
    df.replace('MK Dons','MK Dons', inplace=True)
    df.replace('Mlada Boleslav','Mlada Boleslav', inplace=True)
    df.replace('Mladost Lucani','Mladost', inplace=True)
    df.replace('Modena','Modena', inplace=True)
    df.replace('MOL Vidi','MOL Fehervar', inplace=True)
    df.replace('Molde','Molde', inplace=True)
    df.replace('Molenbeek','RWDM', inplace=True)
    df.replace('Monaco','Monaco', inplace=True)
    df.replace('Monterrey','Monterrey', inplace=True)
    df.replace('Montpellier','Montpellier', inplace=True)
    df.replace('Montrose','Montrose', inplace=True)
    df.replace('Morecambe','Morecambe', inplace=True)
    df.replace('Moreirense','Moreirense', inplace=True)
    df.replace('Morton','Morton', inplace=True)
    df.replace('Moss','Moss', inplace=True)
    df.replace('Motherwell','Motherwell', inplace=True)
    df.replace('MTK Budapest','MTK Budapest', inplace=True)
    df.replace('Mura','Mura', inplace=True)
    df.replace('MVV Maastricht','Maastricht', inplace=True)
    df.replace('NAC Breda','Breda', inplace=True)
    df.replace('Naestved','Naestved', inplace=True)
    df.replace('Nagasaki','V-Varen Nagasaki', inplace=True)
    df.replace('Nagoya','Nagoya Grampus', inplace=True)
    df.replace('Nancy','Nancy', inplace=True)
    df.replace('Nantes','Nantes', inplace=True)
    df.replace('Nantong Zhiyun F.C','Nantong Zhiyun', inplace=True)
    df.replace('Napoli','Napoli', inplace=True)
    df.replace('Nashville SC','Nashville SC', inplace=True)
    df.replace('National Bank','National Bank Egypt', inplace=True)
    df.replace('NEC Nijmegen','Nijmegen', inplace=True)
    df.replace('Necaxa','Necaxa', inplace=True)
    df.replace('Neuchatel Xamax','Xamax', inplace=True)
    df.replace('New England','New England Revolution', inplace=True)
    df.replace('New York City','New York City', inplace=True)
    df.replace('New York Red Bulls','New York Red Bulls', inplace=True)
    df.replace('Newcastle','Newcastle', inplace=True)
    df.replace('Newcastle Jets','Newcastle Jets', inplace=True)
    df.replace('Newells','Newells Old Boys', inplace=True)
    df.replace('Newport County','Newport', inplace=True)
    df.replace('Newtown','Newtown', inplace=True)
    df.replace('Nice','Nice', inplace=True)
    df.replace('Nimes','Nimes', inplace=True)
    df.replace('Niort','Niort', inplace=True)
    df.replace('NK Aluminij','Aluminij', inplace=True)
    df.replace('NK Bravo','Bravo', inplace=True)
    df.replace('NK Celje','Celje', inplace=True)
    df.replace('NK Croatia Zmijavci','Croatia Zmijavci', inplace=True)
    df.replace('NK Dubrava Zagreb','Dubrava', inplace=True)
    df.replace('NK Dugopolje','Dugopolje', inplace=True)
    df.replace('NK Istra','Istra 1961', inplace=True)
    df.replace('NK Jarun','Jarun', inplace=True)
    df.replace('NK Maribor','Maribor', inplace=True)
    df.replace('Nk Posusje','Posusje', inplace=True)
    df.replace('NK Radomlje','Radomlje', inplace=True)
    df.replace('NK Rogaska','Rogaska', inplace=True)
    df.replace('NK Sesvete','Sesvete', inplace=True)
    df.replace('NK Solin','Solin', inplace=True)
    df.replace('Nomme Kalju','Kalju', inplace=True)
    df.replace('Nomme Utd','Nomme Utd', inplace=True)
    df.replace('Norrkoping','Norrkoping', inplace=True)
    df.replace('Northampton','Northampton', inplace=True)
    df.replace('Norwich','Norwich', inplace=True)
    df.replace('Nottm Forest','Nottingham', inplace=True)
    df.replace('Notts Co','Notts Co', inplace=True)
    df.replace('Novorizontino','Novorizontino', inplace=True)
    df.replace('Nurnberg','Nurnberg', inplace=True)
    df.replace('OB','Odense', inplace=True)
    df.replace('Obolon-Brovar Kiev','Obolon', inplace=True)
    df.replace('Odds BK','Odd', inplace=True)
    df.replace('Oita','Oita Trinita', inplace=True)
    df.replace('Okayama','Okayama', inplace=True)
    df.replace('Oldham','Oldham', inplace=True)
    df.replace('Oleksandria','Oleksandriya', inplace=True)
    df.replace('Olimpija','O. Ljubljana', inplace=True)
    df.replace('Oliveirense','Oliveirense', inplace=True)
    df.replace('Omiya','Omiya Ardija', inplace=True)
    df.replace('Orebro','Orebro', inplace=True)
    df.replace('Orgryte','Orgryte', inplace=True)
    df.replace('Orlando City','Orlando City', inplace=True)
    df.replace('Orleans','Orleans', inplace=True)
    df.replace('Osasuna','Osasuna', inplace=True)
    df.replace('Osijek','Osijek', inplace=True)
    df.replace('Osters','Oster', inplace=True)
    df.replace('Ostersunds FK','Ostersund', inplace=True)
    df.replace('Otelul Galati','Otelul', inplace=True)
    df.replace('Oud-Heverlee Leuven','Leuven', inplace=True)
    df.replace('Oviedo','R. Oviedo', inplace=True)
    df.replace('Oxford City','Oxford City', inplace=True)
    df.replace('Oxford Utd','Oxford Utd', inplace=True)
    df.replace('Pachuca','Pachuca', inplace=True)
    df.replace('Pacos Ferreira','Pacos Ferreira', inplace=True)
    df.replace('Paderborn','Paderborn', inplace=True)
    df.replace('Paide Linnameeskond','Paide', inplace=True)
    df.replace('Paide Linnameeskond II','Paide Linnameeskond U21', inplace=True)
    df.replace('Paks','Paks', inplace=True)
    df.replace('Palermo','Palermo', inplace=True)
    df.replace('Pardubice','FK Pardubice', inplace=True)
    df.replace('Paris FC','Paris FC', inplace=True)
    df.replace('Paris St-G','Paris SG', inplace=True)
    df.replace('Parma','Parma', inplace=True)
    df.replace('Parnu JK Vaprus','Parnu JK Vaprus', inplace=True)
    df.replace('Partick','Partick Thistle', inplace=True)
    df.replace('Partizan Belgrade','Partizan', inplace=True)
    df.replace('Patro Eisden Maasmechelen','Patro Eisden', inplace=True)
    df.replace('Pau','Pau FC', inplace=True)
    df.replace('PEC Zwolle','Zwolle', inplace=True)
    df.replace('Penafiel','Penafiel', inplace=True)
    df.replace('Pendikspor','Pendikspor', inplace=True)
    df.replace('Penybont FC','Penybont', inplace=True)
    df.replace('Perth Glory','Perth Glory', inplace=True)
    df.replace('Peterborough','Peterborough', inplace=True)
    df.replace('Peterhead','Peterhead', inplace=True)
    df.replace('PFC Levski Sofia','Levski Sofia', inplace=True)
    df.replace('Pharco FC','Pharco', inplace=True)
    df.replace('Philadelphia','Philadelphia Union', inplace=True)
    df.replace('Piast Gliwice','Piast Gliwice', inplace=True)
    df.replace('Pirin Blagoevgrad','Pirin Blagoevgrad', inplace=True)
    df.replace('Pisa','Pisa', inplace=True)
    df.replace('Plymouth','Plymouth', inplace=True)
    df.replace('Plzen','Plzen', inplace=True)
    df.replace('Podbrezova','Podbrezova', inplace=True)
    df.replace('Pogon Szczecin','Pogon Szczecin', inplace=True)
    df.replace('Pohang Steelers','Pohang', inplace=True)
    df.replace('Polissya Zhytomyr','Zhytomyr', inplace=True)
    df.replace('Ponte Preta','Ponte Preta', inplace=True)
    df.replace('Pontypridd Town','Pontypridd', inplace=True)
    df.replace('Port Vale','Port Vale', inplace=True)
    df.replace('Portimonense','Portimonense', inplace=True)
    df.replace('Portland Timbers','Portland Timbers', inplace=True)
    df.replace('Porto','FC Porto', inplace=True)
    df.replace('Porto B','FC Porto B', inplace=True)
    df.replace('Portsmouth','Portsmouth', inplace=True)
    df.replace('Preston','Preston', inplace=True)
    df.replace('Preussen Munster','Preussen Munster', inplace=True)
    df.replace('PSV','PSV', inplace=True)
    df.replace('Puebla','Puebla', inplace=True)
    df.replace('Pumas UNAM','U.N.A.M.- Pumas', inplace=True)
    df.replace('Puskas Akademia','Puskas Academy', inplace=True)
    df.replace('Puszcza Niepolomice','Puszcza', inplace=True)
    df.replace('Pyramids','Pyramids', inplace=True)
    df.replace('Qingdao Jonoon','Qingdao Hainiu', inplace=True)
    df.replace('QPR','QPR', inplace=True)
    df.replace('Queen of South','Queen of South', inplace=True)
    df.replace("Queens Park','Queen's Park", inplace=True)
    df.replace('Queretaro','Queretaro', inplace=True)
    df.replace('Quevilly Rouen','Quevilly Rouen', inplace=True)
    df.replace('Racing Club','Racing Club', inplace=True)
    df.replace('Racing de Ferrol','Ferrol', inplace=True)
    df.replace('Racing Genk B','Genk U23', inplace=True)
    df.replace('Racing Santander','Racing Santander', inplace=True)
    df.replace('Radnicki Nis','Radnicki Nis', inplace=True)
    df.replace('Radnik Surdulica','Radnik', inplace=True)
    df.replace('Radomiak Radom','Radomiak Radom', inplace=True)
    df.replace('Raith','Raith', inplace=True)
    df.replace('Rakow Czestochowa','Rakow', inplace=True)
    df.replace('Randers','Randers FC', inplace=True)
    df.replace('Rangers','Rangers', inplace=True)
    df.replace('Ranheim IL','Ranheim', inplace=True)
    df.replace('Rapid Bucharest','FC Rapid Bucuresti', inplace=True)
    df.replace('Rapid Vienna','Rapid Vienna', inplace=True)
    df.replace('Raufoss','Raufoss', inplace=True)
    df.replace('Rayo Vallecano','Rayo Vallecano', inplace=True)
    df.replace('RB Leipzig','RB Leipzig', inplace=True)
    df.replace('Reading','Reading', inplace=True)
    df.replace('Real Madrid','Real Madrid', inplace=True)
    df.replace('Real Salt Lake','Real Salt Lake', inplace=True)
    df.replace('Real Sociedad','Real Sociedad', inplace=True)
    df.replace('Red Bull Salzburg','Salzburg', inplace=True)
    df.replace('Red Star','Red Star', inplace=True)
    df.replace('Reggiana','Reggiana', inplace=True)
    df.replace('Reims','Reims', inplace=True)
    df.replace('Rennes','Rennes', inplace=True)
    df.replace('Renofa Yamaguchi','Renofa Yamaguchi', inplace=True)
    df.replace('Rijeka','Rijeka', inplace=True)
    df.replace('Rio Ave','Rio Ave', inplace=True)
    df.replace('River Plate','River Plate', inplace=True)
    df.replace('Rizespor','Rizespor', inplace=True)
    df.replace('RKC Waalwijk','Waalwijk', inplace=True)
    df.replace('Rochdale','Rochdale', inplace=True)
    df.replace('Roda JC','Roda', inplace=True)
    df.replace('Rodez','Rodez', inplace=True)
    df.replace('Roma','AS Roma', inplace=True)
    df.replace('Rosario Central','Rosario Central', inplace=True)
    df.replace('Rosenborg','Rosenborg', inplace=True)
    df.replace('Ross Co','Ross County', inplace=True)
    df.replace('Rotherham','Rotherham', inplace=True)
    df.replace('Rot-Weiss Essen','RW Essen', inplace=True)
    df.replace('Rouen','Rouen', inplace=True)
    df.replace('Ruch Chorzow','Ruch Chorzow', inplace=True)
    df.replace('Rudes','Rudes', inplace=True)
    df.replace('Rukh Vynnyky','Rukh Lviv', inplace=True)
    df.replace('Ruzomberok','Ruzomberok', inplace=True)
    df.replace('Saarbrucken','Saarbrucken', inplace=True)
    df.replace('Sakaryaspor','Sakaryaspor', inplace=True)
    df.replace('Salernitana','Salernitana', inplace=True)
    df.replace('Salford City','Salford', inplace=True)
    df.replace('Salpa','SalPa', inplace=True)
    df.replace('Sampaio Correa FC','Sampaio Correa', inplace=True)
    df.replace('Sampdoria','Sampdoria', inplace=True)
    df.replace('Samsunspor','Samsunspor', inplace=True)
    df.replace('San Jose Earthquakes','San Jose Earthquakes', inplace=True)
    df.replace('San Lorenzo','San Lorenzo', inplace=True)
    df.replace('Sandefjord','Sandefjord', inplace=True)
    df.replace('Sandnes Ulf','Sandnes', inplace=True)
    df.replace('Sanliurfaspor','Sanliurfaspor', inplace=True)
    df.replace('Santa Clara','Santa Clara', inplace=True)
    df.replace('Santos','Santos', inplace=True)
    df.replace('Santos Laguna','Santos Laguna', inplace=True)
    df.replace('Sao Paulo','Sao Paulo', inplace=True)
    df.replace('Sapporo','Hokkaido Consadole Sapporo', inplace=True)
    df.replace('Sarajevo','FK Sarajevo', inplace=True)
    df.replace('Sarmiento de Junin','Sarmiento Junin', inplace=True)
    df.replace('Sarpsborg','Sarpsborg 08', inplace=True)
    df.replace('Sassuolo','Sassuolo', inplace=True)
    df.replace('SC Austria Lustenau','A. Lustenau', inplace=True)
    df.replace('SC Telstar','Telstar', inplace=True)
    df.replace('Schaffhausen','Schaffhausen', inplace=True)
    df.replace('Schalke 04','Schalke', inplace=True)
    df.replace('SCR Altach','Altach', inplace=True)
    df.replace('SE Palmeiras','Palmeiras', inplace=True)
    df.replace('Seattle Sounders','Seattle Sounders', inplace=True)
    df.replace('Sendai','Vegalta Sendai', inplace=True)
    df.replace('Seongnam FC','Seongnam', inplace=True)
    df.replace('Seoul E-Land FC','Seoul E-Land', inplace=True)
    df.replace('Seraing Utd','Seraing', inplace=True)
    df.replace('Servette','Servette', inplace=True)
    df.replace('Sevilla','Sevilla', inplace=True)
    df.replace('Shakhtar','Shakhtar Donetsk', inplace=True)
    df.replace('Shamrock Rovers','Shamrock Rovers', inplace=True)
    df.replace('Shandong Taishan','Shandong Taishan', inplace=True)
    df.replace('Shanghai Port FC','Shanghai Port', inplace=True)
    df.replace('Shanghai Shenhua','Shanghai Shenhua', inplace=True)
    df.replace('Sheff Utd','Sheffield Utd', inplace=True)
    df.replace('Sheff Wed','Sheffield Wed', inplace=True)
    df.replace('Shelbourne','Shelbourne', inplace=True)
    df.replace('Shenzhen FC','Shenzhen', inplace=True)
    df.replace('Shimizu','Shimizu S-Pulse', inplace=True)
    df.replace('Shonan','Shonan Bellmare', inplace=True)
    df.replace('Shrewsbury','Shrewsbury', inplace=True)
    df.replace('Sibenik','Sibenik', inplace=True)
    df.replace('Sigma Olomouc','Sigma Olomouc', inplace=True)
    df.replace('Silkeborg','Silkeborg', inplace=True)
    df.replace('Sint Truiden','St. Truiden', inplace=True)
    df.replace('Sion','Sion', inplace=True)
    df.replace('Sirius','Sirius', inplace=True)
    df.replace('Siroki Brijeg','Siroki Brijeg', inplace=True)
    df.replace('Sivasspor','Sivasspor', inplace=True)
    df.replace('SJK','SJK', inplace=True)
    df.replace('SJK 2','SJK Akatemia', inplace=True)
    df.replace('SK Sturm Graz II','Sturm Graz (Am)', inplace=True)
    df.replace('Skeid','Skeid', inplace=True)
    df.replace('Skovde AIK','Skovde AIK', inplace=True)
    df.replace('SKU Amstetten','Amstetten', inplace=True)
    df.replace('SL 16 FC','St. Liege U23', inplace=True)
    df.replace('Slask Wroclaw','Slask Wroclaw', inplace=True)
    df.replace('Slaven Belupo','Slaven Belupo', inplace=True)
    df.replace('Slavia Prague','Slavia Prague', inplace=True)
    df.replace('Slavia Sofia','Slavia Sofia', inplace=True)
    df.replace('Sligo Rovers','Sligo Rovers', inplace=True)
    df.replace('Sloga Doboj','Sloga Doboj', inplace=True)
    df.replace('Slovacko','Slovacko', inplace=True)
    df.replace('Slovan Bratislava','Slovan Bratislava', inplace=True)
    df.replace('Slovan Liberec','Liberec', inplace=True)
    df.replace('Smouha','Smouha', inplace=True)
    df.replace('Sochaux','Sochaux', inplace=True)
    df.replace('Sogndal','Sogndal', inplace=True)
    df.replace('Solihull Moors','Solihull Moors', inplace=True)
    df.replace('SonderjyskE','Sonderjyske', inplace=True)
    df.replace('Southampton','Southampton', inplace=True)
    df.replace('Southend','Southend', inplace=True)
    df.replace('Sparta Prague','Sparta Prague', inplace=True)
    df.replace('Sparta Rotterdam','Sparta Rotterdam', inplace=True)
    df.replace('Spartak Trnava','Trnava', inplace=True)
    df.replace('Spartans','Spartans', inplace=True)
    df.replace('Spezia','Spezia', inplace=True)
    df.replace('Sport Recife','Sport Recife', inplace=True)
    df.replace('Sporting Gijon','Gijon', inplace=True)
    df.replace('Sporting Lisbon','Sporting CP', inplace=True)
    df.replace('SSD Bari','Bari', inplace=True)
    df.replace('SSV Ulm','Ulm', inplace=True)
    df.replace('St Etienne','St Etienne', inplace=True)
    df.replace('St Gallen','St. Gallen', inplace=True)
    df.replace('St Johnstone','St Johnstone', inplace=True)
    df.replace('St Louis City SC','St. Louis City', inplace=True)
    df.replace('St Mirren','St. Mirren', inplace=True)
    df.replace('St Patricks','St. Patricks', inplace=True)
    df.replace('St Pauli','St. Pauli', inplace=True)
    df.replace('St Polten','St. Polten', inplace=True)
    df.replace('Stabaek','Stabaek', inplace=True)
    df.replace('Stade Lausanne-Ouchy','Lausanne Ouchy', inplace=True)
    df.replace('Stade Nyonnais','Stade Nyonnais', inplace=True)
    df.replace('Stal Mielec','Stal Mielec', inplace=True)
    df.replace('Standard','St. Liege', inplace=True)
    df.replace('Start','Start', inplace=True)
    df.replace('Stenhousemuir','Stenhousemuir', inplace=True)
    df.replace('Stevenage','Stevenage', inplace=True)
    df.replace('Stirling','Stirling', inplace=True)
    df.replace('Stjarnan','Stjarnan', inplace=True)
    df.replace('Stockport','Stockport County', inplace=True)
    df.replace('Stoke','Stoke', inplace=True)
    df.replace('Stranraer','Stranraer', inplace=True)
    df.replace('Strasbourg','Strasbourg', inplace=True)
    df.replace('Stromsgodset','Stromsgodset', inplace=True)
    df.replace('Sturm Graz','Sturm Graz', inplace=True)
    df.replace('Stuttgart','Stuttgart', inplace=True)
    df.replace('Sudtirol','Sudtirol', inplace=True)
    df.replace('Sunderland','Sunderland', inplace=True)
    df.replace('Sundsvall','Sundsvall', inplace=True)
    df.replace('Sutton Utd','Sutton', inplace=True)
    df.replace('Suwon Bluewings','Suwon Bluewings', inplace=True)
    df.replace('Suwon FC','Suwon FC', inplace=True)
    df.replace('SV Darmstadt','Darmstadt', inplace=True)
    df.replace('SV Horn','Horn', inplace=True)
    df.replace('SV Lafnitz','Lafnitz', inplace=True)
    df.replace('SV Ried','Ried', inplace=True)
    df.replace('SV Sandhausen','Sandhausen', inplace=True)
    df.replace('SV Stripfing/Weiden','Stripfing', inplace=True)
    df.replace('Swansea','Swansea', inplace=True)
    df.replace('Swindon','Swindon', inplace=True)
    df.replace('Sydney FC','Sydney FC', inplace=True)
    df.replace('Tabasalu JK','Tabasalu', inplace=True)
    df.replace('Talleres','Talleres Cordoba', inplace=True)
    df.replace('Tallinna FC Flora','Flora', inplace=True)
    df.replace('Tallinna JK Legion','Legion', inplace=True)
    df.replace('Tallinna Kalev','Tallinna Kalev', inplace=True)
    df.replace('Tammeka Tartu','Tammeka', inplace=True)
    df.replace('Tapatio','Tapatio', inplace=True)
    df.replace('Tenerife','Tenerife', inplace=True)
    df.replace('Teplice','Teplice', inplace=True)
    df.replace('Ternana','Ternana', inplace=True)
    df.replace('The New Saints','TNS', inplace=True)
    df.replace('Thespakusatsu Gunma','Kusatsu', inplace=True)
    df.replace('Thun','Thun', inplace=True)
    df.replace('Tianjin Jinmen Tiger FC','Tianjin Jinmen Tiger', inplace=True)
    df.replace('Tigre','Tigre', inplace=True)
    df.replace('Tigres','U.A.N.L.- Tigres', inplace=True)
    df.replace('Tijuana','Club Tijuana', inplace=True)
    df.replace('Tlaxcala F.C','Tlaxcala', inplace=True)
    df.replace('Tochigi SC','Tochigi SC', inplace=True)
    df.replace('Tokushima','Tokushima', inplace=True)
    df.replace('Tokyo-V','Verdy', inplace=True)
    df.replace('Toluca','Toluca', inplace=True)
    df.replace('Tombense MG','Tombense', inplace=True)
    df.replace('Tondela','Tondela', inplace=True)
    df.replace('Torino','Torino', inplace=True)
    df.replace('Toronto FC','Toronto FC', inplace=True)
    df.replace('Torreense','Torreense', inplace=True)
    df.replace('Tosu','Sagan Tosu', inplace=True)
    df.replace('Tottenham','Tottenham', inplace=True)
    df.replace('Toulouse','Toulouse', inplace=True)
    df.replace('TPS','TPS', inplace=True)
    df.replace('Trabzonspor','Trabzonspor', inplace=True)
    df.replace('Tranmere','Tranmere', inplace=True)
    df.replace('Trans Narva','Narva', inplace=True)
    df.replace('Trelleborgs','Trelleborg', inplace=True)
    df.replace('Trencin','Trencin', inplace=True)
    df.replace('Tromso','Tromso', inplace=True)
    df.replace('Tuzla City','Tuzla City', inplace=True)
    df.replace('Tuzlaspor','Tuzlaspor', inplace=True)
    df.replace('UCD','UC Dublin', inplace=True)
    df.replace('Udinese','Udinese', inplace=True)
    df.replace('Ujpest','Ujpest', inplace=True)
    df.replace('Ulsan Hyundai Horang-i','Ulsan Hyundai', inplace=True)
    df.replace('Umraniyespor','Umraniyespor', inplace=True)
    df.replace('Union Berlin','Union Berlin', inplace=True)
    df.replace('Union Santa Fe','Union de Santa Fe', inplace=True)
    df.replace('Union St Gilloise','Royale Union SG', inplace=True)
    df.replace('Universitatea Cluj','U. Cluj', inplace=True)
    df.replace('Universitatea Craiova','Univ. Craiova', inplace=True)
    df.replace('Unterhaching','Unterhaching', inplace=True)
    df.replace('Urawa','Urawa Reds', inplace=True)
    df.replace('US Cremonese','Cremonese', inplace=True)
    df.replace('UTA Arad','UTA Arad', inplace=True)
    df.replace('Utsiktens','Utsikten', inplace=True)
    df.replace('Valencia','Valencia', inplace=True)
    df.replace('Valenciennes','Valenciennes', inplace=True)
    df.replace('Valerenga','Valerenga', inplace=True)
    df.replace('Valladolid','Valladolid', inplace=True)
    df.replace('Valur','Valur', inplace=True)
    df.replace('Vancouver Whitecaps','Vancouver Whitecaps', inplace=True)
    df.replace('Varazdin','Varazdin', inplace=True)
    df.replace('Varbergs BoIS','Varberg', inplace=True)
    df.replace('Varnamo','Varnamo', inplace=True)
    df.replace('Vasco da Gama','Vasco', inplace=True)
    df.replace('Vasteras SK','Vasteras SK', inplace=True)
    df.replace('Vejle','Vejle', inplace=True)
    df.replace('Velez Sarsfield','Velez Sarsfield', inplace=True)
    df.replace('Venados FC','Venados', inplace=True)
    df.replace('Vendsyssel FF','Vendsyssel', inplace=True)
    df.replace('Venezia','Venezia', inplace=True)
    df.replace('Veres Rivne','Veres-Rivne', inplace=True)
    df.replace('Verl','Verl', inplace=True)
    df.replace('Verona','Verona', inplace=True)
    df.replace('Versailles 78 FC','Versailles', inplace=True)
    df.replace('VfL Osnabruck','VfL Osnabruck', inplace=True)
    df.replace('Viborg','Viborg', inplace=True)
    df.replace('Viimsi JK','Viimsi JK', inplace=True)
    df.replace('Viking','Viking', inplace=True)
    df.replace('Vikingur Reykjavik','Vikingur Reykjavik', inplace=True)
    df.replace('Viktoria Koln','Viktoria Koln', inplace=True)
    df.replace('Vila Nova','Vila Nova FC', inplace=True)
    df.replace('Vilaverdense','Vilaverdense', inplace=True)
    df.replace('Villarreal','Villarreal', inplace=True)
    df.replace('Villarreal B','Villarreal B', inplace=True)
    df.replace('Villefranche Beaujolais','Villefranche', inplace=True)
    df.replace('Vitesse Arnhem','Vitesse', inplace=True)
    df.replace('Vizela','Vizela', inplace=True)
    df.replace('Vojvodina','Vojvodina', inplace=True)
    df.replace('Vorskla','Vorskla Poltava', inplace=True)
    df.replace('Vozdovac','Vozdovac', inplace=True)
    df.replace('VPS','VPS', inplace=True)
    df.replace('Vukovar','Vukovar 1991', inplace=True)
    df.replace('VVV Venlo','Venlo', inplace=True)
    df.replace('Waasland-Beveren','Beveren', inplace=True)
    df.replace('Waldhof Mannheim','Mannheim', inplace=True)
    df.replace('Walsall','Walsall', inplace=True)
    df.replace('Warta Poznan','Warta Poznan', inplace=True)
    df.replace('Watford','Watford', inplace=True)
    df.replace('Wealdstone','Wealdstone', inplace=True)
    df.replace('Wehen Wiesbaden','Wehen', inplace=True)
    df.replace('Wellington Phoenix','Wellington Phoenix', inplace=True)
    df.replace('Werder Bremen','Werder Bremen', inplace=True)
    df.replace('West Armenia','West Armenia', inplace=True)
    df.replace('West Brom','West Brom', inplace=True)
    df.replace('West Ham','West Ham', inplace=True)
    df.replace('Westerlo','Westerlo', inplace=True)
    df.replace('Western Sydney Wanderers','WS Wanderers', inplace=True)
    df.replace('Western United','Western United', inplace=True)
    df.replace('Widzew Lodz','Widzew Lodz', inplace=True)
    df.replace('Wigan','Wigan', inplace=True)
    df.replace('Willem II','Willem II', inplace=True)
    df.replace('Winterthur','Winterthur', inplace=True)
    df.replace('Woking','Woking', inplace=True)
    df.replace('Wolfsberger AC','Wolfsberger AC', inplace=True)
    df.replace('Wolfsburg','Wolfsburg', inplace=True)
    df.replace('Wolves','Wolves', inplace=True)
    df.replace('Wrexham','Wrexham', inplace=True)
    df.replace('WSG Wattens','Tirol', inplace=True)
    df.replace('Wuhan Three Towns','Wuhan Three Towns', inplace=True)
    df.replace('Wycombe','Wycombe', inplace=True)
    df.replace('Yamagata','Montedio Yamagata', inplace=True)
    df.replace('Yellow-Red Mechelen','KV Mechelen', inplace=True)
    df.replace('Yokohama FC','Yokohama FC', inplace=True)
    df.replace('Yokohama FM','Yokohama F. Marinos', inplace=True)
    df.replace('York City','York City', inplace=True)
    df.replace('Young Boys','Young Boys', inplace=True)
    df.replace('Yverdon Sport','Yverdon', inplace=True)
    df.replace('Zaglebie Lubin','Zaglebie', inplace=True)
    df.replace('Zalaegerszeg','Zalaegerszegi', inplace=True)
    df.replace('Zamalek','Zamalek', inplace=True)
    df.replace('Zaragoza','Zaragoza', inplace=True)
    df.replace('ZED FC','ZED', inplace=True)
    df.replace('Zeleznicar Pancevo','Zeleznicar Pancevo', inplace=True)
    df.replace('Zeljeznicar','Zeljeznicar', inplace=True)
    df.replace('Zemplin','Michalovce', inplace=True)
    df.replace('Zhejiang Greentown','Zhejiang Professional', inplace=True)
    df.replace('Zilina','Zilina', inplace=True)
    df.replace('Zlate Moravce','Z. Moravce-Vrable', inplace=True)
    df.replace('Zlin','Zlin', inplace=True)
    df.replace('Zorya','FK Zorya Luhansk', inplace=True)
    df.replace('Zrinjski','Zrinjski', inplace=True)
    df.replace('Zrinski Jurjevac','Zrinski Jurjevac', inplace=True)
    df.replace('Zulte-Waregem','Waregem', inplace=True)
    df.replace('Zvijezda 09 Bijeljina','Zvijezda 09', inplace=True)
    
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
    
def grafico(dataframe,target):
    dataframe = dataframe.copy()
    dataframe.reset_index(drop=True,inplace=True)
    dataframe['Target_Acum'] = dataframe[target].cumsum()
    roi = dataframe[target].sum()/dataframe.shape[0]
    profit = dataframe[target].sum()
    dataframe['Target_Acum'].plot(title=f'ROI = {round(roi*100,2)} % \nProfit = {round(profit,2)} Stakes',xlabel=f'{dataframe.shape[0]} Entradas')



