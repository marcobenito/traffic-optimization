import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from math import sqrt
from datetime import datetime

import nasch
plt.figure(figsize=(6.5,5))
for year in [2019]:
    sensor = 6694
    final = pd.read_csv(str(year) + '.csv', delimiter=';')
    print(final.info())
    sensors_path = 'Datos_puntos_de_medida_M-30.xlsx'
    sensors = xlrd.open_workbook(sensors_path).sheet_by_index(0)
    interior_s_id = [sensors.cell_value(row, 1) for row in range(5,61)]
    interior_s_pos = [sensors.cell_value(row, 3) for row in range(5,61)]


    hola = final[(final['id'] == sensor) & (final['tipo_dia'] == 'Diario')]

    plt.scatter(hola['ocupacion'], hola['intensidad'], label=str(year))
    # plt.scatter(hola['ocupacion'], hola['vmed'])
plt.title('Ocupacion vs intensidad en {}; sensor {} (I7)'.format(year, sensor))
plt.xlabel('Ocupacion [%]')
plt.ylabel('Intensidad [veh/h]')
plt.legend()
# plt.show()


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


### PRINT INTENSITY VS OCCUPATION FOR DIFFERENT WEEK DAYS

# ids = [5423]
days = [['-02-03', '-02-10', '-02-17', '-02-24'],
        ['-02-04', '-02-11', '-02-18', '-02-25'],
        ['-02-05', '-02-12', '-02-19', '-02-26'],
        ['-02-06', '-02-13', '-02-20', '-02-27'],
        ['-02-07', '-02-14', '-02-21', '-02-28'],
        ['-02-08', '-02-15', '-02-22', '-02-29', '-02-01'],
        ['-02-09', '-02-16', '-02-23', '-02-02']]

sensor = 6640
### CALL TO NASCH FUNCTION TO OBTAIN CURVES FOR DIFFERENT VALUES OF 'P'

n = 150       #number of space positions
t = 150          #number of space intervals
vmax = 6            #Maximum road speed
ntl = 1            #number of traffic lights in the road

entrance_positions = np.array([0, 0])
entrance_probability = np.array([0, 0])
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
                rep = 20
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

# plt.figure()
sensores = [3816, 6672, 6674]
sensores = [6648, 6650, 10297]
sensores = [6640,6642,6644]
# sensores = [6644]
dias = ['Diario','Sabado','Festivo']
# for dia in dias:
for sensor in sensores:
    data = final[(final['id'] == sensor) & (final['tipo_dia'] == 'Diario')]
    plt.figure(10)
    plt.scatter(data['ocupacion'], data['carga'], label='Sensor {}'.format(sensor))
    plt.figure(11)
    plt.scatter(data['ocupacion'], data['intensidad'], label='Sensor {}'.format(sensor))
plt.legend()
# plt.title('Sensor {} in 2019'.format(sensor))
plt.xlabel('ocupacion [%]')
plt.ylabel('intensidad [veh/h]')
# plt.show()


months=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Sep', 'Oct', 'Nov', 'Dec']
plt.figure(100)
for sensor in sensores:
    data = final[(final['id'] == sensor) & (final['mes'].isin(months)) & (final['tipo_dia'] == 'Diario')]
    data1 = final[(final['id'] == sensor) & (final['dia'] == '2020-02-25')]
    hours = pd.unique(data['hora'])
    hours = hours[24:80]
    std, mean = np.zeros(len(hours)), np.zeros(len(hours))
    for i, hour in enumerate(hours):
        data1 = data[data['hora'] == hour]
        mean[i] = data1['intensidad'].mean()
        std[i] = data1['intensidad'].std()
        # mean[i] = max(data1['ocupacion'])
        # std[i] = max(data1['ocupacion'])

    # plt.plot(hours, mean, label='Sensor: {}'.format(sensor))
    # plt.plot(hours, mean)
    plt.plot(hours, data1['intensidad'].iloc[24:80], label='Sensor: {}'.format(sensor))
    # plt.fill_between(hours,mean+std, mean-std, alpha=0.2, label='Sensor: {}'.format(sensor))
# plt.xticks(['00:00','03:45', '07:30', '11:15', '15:00' ,'19:15','23:00'], rotation=90)
plt.xticks(rotation=45)
plt.legend()
plt.xlabel('Hora')
plt.ylabel('Intensidad [veh/h]')
# plt.show()


days = ['10','11','12','13','14']
plt.figure()
for day in days:
    t_day = '2019-02-'+day
    nnn = final[(final['hora'] == '18:00') & (final['dia'] == t_day)]
    # for i in range(len(nnn['vmed'])):
        # if nnn['vmed'].iloc[i]<20:
        #     nnn['vmed'].iloc[i] = 20
    plt.plot(nnn['posicion'], nnn['vmed'], label=t_day)
plt.xlabel('posicion [km]')
plt.ylabel('Velocidad media [km/h]')
plt.legend()
plt.show()

