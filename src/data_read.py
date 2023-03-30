import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from math import sqrt
from datetime import datetime

import nasch

final = pd.read_csv('D:/Documentos/Prácticas Tráfico/M30/Historicos/years/2019.csv', delimiter=';')
print(final.info())
sensors_path = 'D:/Documentos/Prácticas Tráfico/M30/Anexos/Datos_puntos_de_medida_M-30.xlsx'
sensors = xlrd.open_workbook(sensors_path).sheet_by_index(0)
interior_s_id = [sensors.cell_value(row, 1) for row in range(5,61)]
interior_s_pos = [sensors.cell_value(row, 3) for row in range(5,61)]

# def sensor_pos(id):
#     idx = interior_s_id.index(id)
#     return interior_s_pos[idx]
#
# def vel_in_nasch(v):
#     return v / 15.12
#
# def int_to_dens(serie):
#     i = serie[4]
#     v = serie[10]
#     try:
#         return i / (3600 * v) * 100
#     except:
#         return 0
#
# def dates_to_day(date):
#     return date.split(' ')[0]
#
# def dates_to_hour(date):
#     return date.split(' ')[1][:-3]
#
# def dates_to_month(date):
#     """Gives the abbreviated name of the month from a given date
#     The date must be given in the following format: yyyy-mm-dd.
#     Example:
#     given the date 2020-02-05 will return 'Feb'
#     :param date: the date from which to obtain the month
#     :return the abbreviated name of the month"""
#     day = date.split(' ')[0]
#     dt_object = datetime.strptime(day, '%Y-%m-%d')
#
#     return dt_object.strftime('%b')
#
#
# def dates_to_daytype(date):
#     """Gives the type of the day {Diario, Sabado, Festivo}
#        The date must be given in the following format: yyyy-mm-dd.
#        Example:
#        given the date 2020-02-05 will return 'Diario'
#        :param date: the date from which to obtain the day type
#        :return the type of the day"""
#     daytype= {'Mon': 'Diario', 'Tue': 'Diario',
#               'Wed': 'Diario', 'Thu': 'Diario',
#               'Fri': 'Diario', 'Sat': 'Sabado',
#               'Sun': 'Festivo'}
#     day = date.split(' ')[0]
#     dt_object = datetime.strptime(day, '%Y-%m-%d')
#
#     return (daytype[dt_object.strftime('%a')])
#
#
# # Some useful values are added to the dataframe as new columns. For instance,
# # the mean velocity transformed into NaSch units, density, position of the
# # sensors in the road, or date split into day and hour for the ease of consulting
# final['v_nasch'] = final['vmed'].apply(vel_in_nasch)
# final['densidad'] = final.apply(int_to_dens, axis=1)
# final['dia'] = final['fecha'].apply(dates_to_day)
# final['hora'] = final['fecha'].apply(dates_to_hour)
# final['posicion'] = final['id'].apply(sensor_pos)
# final['mes'] = final['fecha'].apply(dates_to_month)
# final['tipo_dia'] = final['fecha'].apply(dates_to_daytype)
#
# # The values are sorted by the position of the sensors in the road
# final.sort_values(by=['posicion', 'hora'], inplace=True)
#
# print(final.head(15))
# print(final.info())
# print(final['id'].describe())
#
# print(final[(final['fecha'].str.contains('2020-02-03')) &
#             (final['id'] == 6687)].head())
# hola = final[(final['fecha'].str.contains('2020-02-08')) & (final['id'] == 6640)]

# hola = final[(final['fecha'].str.contains('2020-02-03'))]
hola = final[(final['id'] == 6640) & (final['dia'].str.contains('2019-02-19'))]
print(hola[['id', 'tipo_dia', 'dia', 'hora' ,'ocupacion']].head(78))


def std_error(points, curve):
    """Computes the standard error of an estimated curve.
    :param points: List of pair of points [x, y].
    :param curve: estimated curve.
    :return error: the total error of the estimation.
    """
    x = np.asarray(points[0])
    y = np.asarray(points[1])
    error = sqrt(sum((y - curve(x))**2) / len(x))

    return error


### PRINT INTENSUTY VS OCCUPATION FOR DIFFERENT WEEK DAYS

# ids = [5423]
days = [['-02-03', '-02-10', '-02-17', '-02-24'],
        ['-02-04', '-02-11', '-02-18', '-02-25'],
        ['-02-05', '-02-12', '-02-19', '-02-26'],
        ['-02-06', '-02-13', '-02-20', '-02-27'],
        ['-02-07', '-02-14', '-02-21', '-02-28'],
        ['-02-08', '-02-15', '-02-22', '-02-29', '-02-01'],
        ['-02-09', '-02-16', '-02-23', '-02-02']]
# aa = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes','sabado', 'domingo']
#
# plt.figure()
# for id in ids:
#     for idx, week_day in enumerate(days):
#         ss1 = final[(final['id'] == id) & (final['dia'].isin(week_day))]
#         plt.scatter(ss1['ocupacion'], ss1['intensidad'], label=aa[idx])
# plt.ylabel('intensidad [veh/h]')
# plt.xlabel('ocupacion [%]')
# plt.legend()
# plt.show()


### PRINT INTENSITY OR OCCUPATION VS HOURS

# hours = ['18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45']
# # ss = final[(final['id'] == 6440) & (final['dia'] == '2020-02-05')]
# ids=[6656, 10175, 3818, 3820]
# for id in ids:
#     ss = final[(final['id'] == id) & (final['dia'] == '2019-02-06') & (final['hora'].isin(hours))]
#     print(ss[['id', 'hora', 'ocupacion', 'intensidad', 'posicion']])
#     plt.plot(ss['hora'], ss['intensidad'], label='id = ' + str(id))
# plt.xlabel('horas')
# plt.ylabel('intensidad')
# plt.legend()
# plt.show()

sensor = 6708
### CALL TO NASCH FUNCTION TO OBTAIN CURVES FOR DIFFERENT VALUES OF 'P'

n = 100       #number of space positions
t = 75          #number of space intervals
vmax = 6            #Maximum road speed
ntl = 1            #number of traffic lights in the road

entrance_positions = np.array([0, 20])
entrance_probability = np.array([0, 0.5])
exit_positions = np.array([0, 20])
exit_probability = np.array([0, 0])

entrance = np.vstack((entrance_positions, entrance_probability, np.zeros(entrance_positions.size)))
exits = np.vstack((exit_positions, exit_probability, np.zeros(exit_positions.size)))

ss = final[(final['id'] == sensor) & (final['tipo_dia'] == 'Diario')]
print('Ahora vamos')
print(ss.head())
def call(fun, *args, **kwargs):
    return fun(*args, *kwargs)

def main(P, P0, density):
    A = nasch.initial_scenario(t, n, density)
    ITL = nasch.initialize_trafficlight(ntl,n)
    # A, means = call(nasch.roundabout, A, P, vmax, ITL,entrance, exit)
    A, means = nasch.straightroad(A, ITL, n, t, density, P, P0, vmax,
                                  entrance=entrance, flag_entry=True)
    # A, means = call(nasch.straightroad, A, ITL, n, t, density,
    #                 P, P0, vmax, flag_entry=True, entrance=entrance, exits=None)
    return A, means[0]

def find_p_p0():
    k = 3
    pp1 = np.linspace(0,0.1,k)
    pp2 = np.array([0.15,0.3, 0.45])
    pp = np.hstack([pp1,pp2])
    P0 = np.linspace(0, 0.6, k)
    density = np.linspace(0.01,1,50)
    means_v = np.zeros((len(pp), len(P0), 50))
    means_f = np.zeros_like(means_v)
    curve, p_store, p0_store = [], [], []
    for i, p in enumerate(pp):
        print(p)
        for ii, p0 in enumerate(P0):
            for j, dens in enumerate(density):
                mean_speed = 0
                rep = 30
                for _ in range(rep):
                    try:
                        mean_speed += main(p, p0, dens)[1]
                    except ZeroDivisionError:
                        pass
                mean_speed /= rep
                means_v[i, ii, j] = mean_speed*18
                means_f[i, ii, j] = mean_speed*dens*3600*1

            estimation = np.polyfit(density*100, means_f[i, ii, :], 8)
            p_store.append(p)
            p0_store.append(p0)
            curve.append(np.poly1d(estimation))

    print(len(curve))
    error_p = np.zeros(len(curve))
    for i in range(len(curve)):

        # error_p[i] = std_error([[i /100 for i in ss['ocupacion']],
        #                         ss['intensidad']], curve[i])
        error_p[i] = std_error([ss['ocupacion'],
                                ss['intensidad']], curve[i])

    # The selected P parameter will be he one which yields less error
    p_idx = np.argmin(error_p)
    P = p_store[p_idx]
    P0 = p0_store[p_idx]
    print('EL valor final de P es: ', P)
    print('EL valor final de P0 es: ', P0)

    dens = np.linspace(min(ss['ocupacion']), max(ss['ocupacion']), 50)
    plt.figure()
    plt.plot(dens, curve[p_idx](dens), color='red',
             label='NaSch (P = {:.2f}, P0 = {:.2f})'.format(P,P0))
    plt.scatter(ss['ocupacion'], ss['intensidad'], label='Datos reales')
    plt.title('Sensor {} vs modelo NaSch'.format(sensor))
    plt.xlabel('ocupacion [%]')
    plt.ylabel('intensidad [veh/h]')
    plt.legend()


    plt.figure()
    for i in range(len(curve)):
    # plt.plot(density*100, means_v[p_idx,:])
        plt.plot(dens, curve[i](dens))#,means_f[i, :])
        # plt.plot(density*100, means_f[i, ii, :])
    plt.scatter(ss['ocupacion'], ss['intensidad'])

    plt.show()

if __name__ == '__main__':
    find_p_p0()


sensores = [6703, 6704, 6708]
dias = ['Diario','Sabado','Festivo']
# for dia in dias:
for sensor in sensores:
    data = final[(final['id'] == sensor) & (final['tipo_dia'] == 'Diario')]
    plt.scatter(data['ocupacion'], data['intensidad'], label='Sensor {}'.format(sensor))
plt.legend()
# plt.title('Sensor {} in 2019'.format(sensor))
plt.xlabel('ocupacion [%]')
plt.ylabel('intensidad [veh/h]')
# plt.show()

months=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Sep', 'Oct', 'Nov', 'Dec']
plt.figure()
for sensor in sensores:
    data = final[(final['id'] == sensor) & (final['mes'].isin(months)) & (final['tipo_dia'] == 'Diario')]
    hours = pd.unique(data['hora'])
    hours = hours[24:80]
    std, mean = np.zeros(len(hours)), np.zeros(len(hours))
    for i, hour in enumerate(hours):
        data1 = data[data['hora'] == hour]
        mean[i] = data1['intensidad'].mean()
        std[i] = data1['intensidad'].std()

    # plt.plot(hours, mean, label='Sensor: {}'.format(sensor))
    plt.plot(hours, mean)
    plt.fill_between(hours,mean+std, mean-std, alpha=0.2, label='Sensor: {}'.format(sensor))
plt.xticks(rotation=45)
plt.legend()
plt.xlabel('Hora')
plt.ylabel('Intensidad [veh/h]')
plt.show()










#


# # hola = final[final['hora'] == '18:00']
#
# # hola['intensidad'].plot(kind='hist');
# hola.plot(kind='scatter', x='posicion', y='densidad', rot='90');
# # hola.plot(kind='line', x='posicion', y='vmed', rot='90');
# # hola1.plot(kind='line', x='posicion', y='vmed', rot='90');
# hola.plot(kind='scatter', x='ocupacion', y='vmed')
# hola.plot(kind='scatter', x='densidad', y='vmed')
#
# # print(final.head())
# # print(hola.head())
# # print(hola['hora'])
# fig, ax1 = plt.subplots()
# ax1.plot(hola['hora'], hola['ocupacion'], label='ocupacion')
# ax1.set_xlabel('hora')
# ax1.tick_params('x', labelrotation=90)
# ax1.set_ylabel('Ocupacion [%]')
# ax2 = ax1.twinx()   #Define a new ax with the same x axis
# ax2.set_ylabel('Mean speed [m/s]')
# ax2.plot(hola['hora'], hola['vmed'], label='vmed', color='red')
# fig.legend()
# fig.tight_layout()  # Otherwise, the y axis will be a little displaced
# ax1.xaxis.set_major_locator(plt.MaxNLocator(7)) # 7 ticks in the x label at most
# plt.title('Ocupacion y velocidad media durante todo el día 03-02-2020, sensor 6644 (km 1.1)')
#
#
#
# # days = ['06', '13', '20', '27']
# #
# # plt.figure()
# # for day in days:
# #     plt.plot(hola[hola['dia'] == '2020-02-' + day]['posicion'],
# #              hola[hola['dia'] == '2020-02-' + day]['ocupacion'], label = day + '-02')
# #
# # plt.title('Mean speed at all sensors every monday in february 2020 at 18:00')
# # plt.xlabel("Sensor's position [km]")
# # plt.ylabel('Mean speed [km/h]')
# # plt.legend()
#
#
# # hola.plot(kind='scatter', x='posicion', y='intensidad', rot='90');
# # hola.plot(kind='scatter', x='posicion', y='v_nasch', rot='90');
# # hola.plot(kind='scatter', x='posicion', y='carga', rot='90');
# plt.show()

# print(hola.posicion.values)  # Returns a numpy array with the values

