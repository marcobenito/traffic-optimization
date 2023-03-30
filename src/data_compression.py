import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
import csv
import os
from datetime import datetime



file = 'D:/Documentos/Prácticas Tráfico/M30/Historicos/02-2019/02-2019.csv'
sensors_path = 'D:/Documentos/Prácticas Tráfico/M30/Anexos/Datos_puntos_de_medida_M-30.xlsx'

sensors = xlrd.open_workbook(sensors_path)
sensors = sensors.sheet_by_index(0)

# interior_s = [(sensors.cell_value(row, 1), sensors.cell_value(row, 3)) for row in range(5,61)]
# entry_s = [(sensors.cell_value(row, 6), sensors.cell_value(row, 8)) for row in range(5, 35)]
interior_s = [sensors.cell_value(row, 1) for row in range(5,62)]
interior_s_pos = [sensors.cell_value(row, 3) for row in range(5,62)]
entry_s = [sensors.cell_value(row, 11) for row in range(5, 31)]
entry_s_pos = [sensors.cell_value(row, 13) for row in range(5, 31)]
exit_s = [sensors.cell_value(row,6) for row in range(5, 34)]
exit_s_pos = [sensors.cell_value(row,8) for row in range(5, 34)]
sensors = interior_s + entry_s + exit_s
position = interior_s_pos + entry_s_pos + exit_s_pos



def sensor_pos(id):
    idx = sensors.index(id)
    return position[idx]

def sensor_type(id):
    if id in interior_s:
        return 'INT'
    elif id in entry_s:
        return 'IN'
    elif id in exit_s:
        return 'OUT'

def dates_to_day(date):
    return date.split(' ')[0]

def dates_to_hour(date):
    return date.split(' ')[1][:-3]

def dates_to_month(date):
    """Gives the abbreviated name of the month from a given date
    The date must be given in the following format: yyyy-mm-dd.
    Example:
    given the date 2020-02-05 will return 'Feb'
    :param date: the date from which to obtain the month
    :return the abbreviated name of the month"""
    day = date.split(' ')[0]
    dt_object = datetime.strptime(day, '%Y-%m-%d')

    return dt_object.strftime('%b')


def dates_to_daytype(date):
    """Gives the type of the day {Diario, Sabado, Festivo}
       The date must be given in the following format: yyyy-mm-dd.
       Example:
       given the date 2020-02-05 will return 'Diario'
       :param date: the date from which to obtain the day type
       :return the type of the day"""
    daytype= {'Mon': 'Diario', 'Tue': 'Diario',
              'Wed': 'Diario', 'Thu': 'Diario',
              'Fri': 'Diario', 'Sat': 'Sabado',
              'Sun': 'Festivo'}
    day = date.split(' ')[0]
    dt_object = datetime.strptime(day, '%Y-%m-%d')

    return (daytype[dt_object.strftime('%a')])

def join_files(path, year):
    path = path + year + '/'
    files = os.listdir(path)
    print(files[0])
    df = pd.read_csv(path + files[0], delimiter=';')
    files.pop(0)
    for file in files:
        print(file)
        df = df.append(pd.read_csv(path + file, delimiter=';'))

    return df


def select_data(file):
    """creates a new dataframe containing all the needed sensors (exit,
    entrance and interior ones) from one and exports it into a new csv file.
    :param file: the monthly file from which to extract the dada"""
    n_chunk = 10 ** 6
    s=0
    df = {}
    for chunk in pd.read_csv(file, delimiter=';', chunksize=n_chunk):
        print('s: ', s)
        # print(chunk.head(), chunk.info()) # Shows general info about the columns (like nº of items, type...)
        # print(chunk.isnull().sum())   # Shows how many NaNs there are in every columns
        chunk.dropna(inplace=True)  # Eliminates the rows containing a NaN
        # print(chunk[['id','vmed']].head())    # Shows a dataframe only with the columns id and vmed
        # print(chunk.iloc[10:20])  # The rows 10:20 selected by index. Using loc selects by index name
        # print(chunk[chunk['vmed'] > 55.0].head())   # SELECT statements are made up of pythonic conditionals
        # print(chunk[chunk['id'].isin([1001,1002])]) # isin method allows to select from multiple names
        # print(chunk['vmed'].describe())   # Gives numerical info about the columns (mean, std, 25%, 50% ...)
        # print(chunk[((chunk['id'] > 1001) & (chunk['id'] < 1004)) & (chunk['vmed'] > 55.0)])  # For and, or statements,
        # also '&' and '|' can be used
        # print(chunk['vmed'].apply(lambda x: 'fast' if x >= 55.0 else 'slow').head())   # With the apply method a function can be
        # be passed to all the values in the column, and a new column will be created with the returned value
        # print(chunk[chunk['tipo_elem'] == 'M30']['id'].value_counts())

        # A new key of the dictionary is created with the values in the chunk that
        # correspond to entry sensors
        if 'idelem' in chunk.columns:
            chunk.rename(columns={'idelem': 'id'}, inplace=True)

        df[str(s)] = chunk[chunk['id'].isin(interior_s+entry_s+exit_s)]
        s+=1

    # The all the empty chunks id's are stored into idx, and then deleted
    idx = [key for key in df.keys() if df[key].shape[0] == 0]
    for i in idx:
        del df[i]

    # A new dataframe is created from the first chunk in the dictionary
    final = df['0']
    # Then it's deleted from the dictionary, and all the other chunks in the
    # dictionary are appended to it.
    del df['0']
    for key in df.keys():
        final = final.append(df[key])

    return final

if __name__ == '__main__':
    root = 'D:/Documentos/Prácticas Tráfico/M30/Historicos/'
    years = ['2015','2016','2017','2018','2019','2020']

    for year in years:
        final = join_files(root, year)
        new_file = root + 'years/' + year + '.csv'
        final.to_csv(new_file, index=False, sep=';')



if __name__ == '__main1__':

    root ='D:/Documentos/Prácticas Tráfico/M30/Historicos/raw_data/'
    years = ['2017']

    for year in years:
        path = root + year + '/'
        files = os.listdir(path)

        for file in files:
            print(file)
            try:
                final = select_data(path + file)

                final['hora'] = final['fecha'].apply(dates_to_hour)
                final['tipo_dia'] = final['fecha'].apply(dates_to_daytype)
                final['dia'] = final['fecha'].apply(dates_to_day)
                final['mes'] = final['fecha'].apply(dates_to_month)
                final['posicion'] = final['id'].apply(sensor_pos)
                final['tipo_sensor'] = final['id'].apply(sensor_type)
                final.drop(['fecha', 'tipo_elem', 'error', 'periodo_integracion'],
                           axis=1, inplace=True)
                if 'tipo' in final.columns:
                    final.drop(['tipo'], axis=1, inplace=True)
                if 'identif' in final.columns:
                    final.drop(['identif'], axis=1, inplace=True)

                #
                final.sort_values(by=['posicion', 'hora'], inplace=True)

                new_file = ''
                file1 = path + file
                for a in file1.split('raw_data/'):
                    new_file += a

                # final.to_csv('C:/Users/marco/Desktop/Prácticas Tráfico/M30/Historicos/02-2019/02-2019-int.csv')
                final.to_csv(new_file, index=False, sep=';')

            except KeyError:
                pass


