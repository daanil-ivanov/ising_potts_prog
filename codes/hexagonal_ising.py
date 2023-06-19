import shutil
import numpy as np
from numpy.random import rand
import matplotlib
import matplotlib.pyplot as plt
import time
import multiprocessing
import os
from datetime import datetime
from scipy.optimize import curve_fit
import zipfile

matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{palatino} \usepackage{textcomp}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']

n_proc = multiprocessing.cpu_count()  # подсчёт логических операторов на компьютере
eqSteps = int(1e3)  # количество МК шагов для прихода в равновесное состояние (рекомендую 300)
mcSteps = int(2e3)  # количество МК шагов для подсчёта физических величин (рекомендую 1000)
it = 300  # количество желаемых температурных точек (рекомендую 100)
calc = it // n_proc + ((it // n_proc) != (it / n_proc))  # количество температурных точек на один логический процессор
nt = int(calc * n_proc)
NN_2 = [50]
NN_3 = [20, 30, 40, 50]
D = [2]
T_2 = np.linspace(1.35, 1.5, nt)
T_3 = np.linspace(2.4, 2.9, nt)

def mcmove_2d(config, beta, N):
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N - 1)
            s = config[a, b]

            if b % 2 == 0:
                if a % 2 == 0:
                    nb = config[(a + 1) % N, b] + config[a, (b + 1) % N] + config[a, (b - 1) % N]
                    # config[(a + 1) % N, (b+1) % N] + config[(a - 1) % N, (b-1) % N]
                else:
                    nb = config[(a - 1) % N, b] + config[a, (b + 1) % N] + config[a, (b - 1) % N]
                    # config[(a + 1) % N, (b+1) % N] + config[(a - 1) % N, (b-1) % N]
            else:
                if i % 2 == 0:
                    nb = config[(a - 1) % N, b] + config[a, (b + 1) % N] + config[a, (b - 1) % N]
                    # config[(a + 1) % N, (b+1) % N] + config[(a - 1) % N, (b-1) % N]
                else:
                    nb = config[(a + 1) % N, b] + config[a, (b + 1) % N] + config[a, (b - 1) % N]
                    # config[(a + 1) % N, (b+1) % N] + config[(a - 1) % N, (b-1) % N]
            cost = 2 * s * nb

            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * beta):
                s *= -1
            config[a, b] = s
    return config


def mcmove_3d(config, beta, N):
    """
    Данная функция позволяет провести один шаг моделирования Монте-Карло
    для трёхмерной решётки методом Метрополиса.
    """
    for i in range(N):
        for j in range(N):
            for k in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                c = np.random.randint(0, N)
                s = config[a, b, c]

                if b % 2 == 0:
                    if a % 2 == 0:
                        nb = config[(a + 1) % N, b, c] + config[a, (b + 1) % N, c] + config[
                            a, (b - 1) % N, c] + \
                             config[a, b, (c + 1) % N] + config[a, b, (c - 1) % N]
                    else:
                        nb = config[(a - 1) % N, b, c] + config[a, (b + 1) % N, c] + config[
                            a, (b - 1) % N, c] + \
                             config[a, b, (c + 1) % N] + config[a, b, (c - 1) % N]
                else:
                    if i % 2 == 0:
                        nb = config[(a - 1) % N, b, c] + config[a, (b + 1) % N, c] + config[
                            a, (b - 1) % N, c] + \
                             config[a, b, (c + 1) % N] + config[a, b, (c - 1) % N]
                    else:
                        nb = config[(a + 1) % N, b, c] + config[a, (b + 1) % N, c] + config[
                            a, (b - 1) % N, c] + \
                             config[a, b, (c + 1) % N] + config[a, b, (c - 1) % N]
                cost = 2 * s * nb

                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost * beta):
                    s *= -1
                config[a, b] = s
    return config


def calcEnergy_2d(config, N):
    """
    Energy of a given configuration
    """
    energy = 0

    for i in range(N):
        for j in range(N):
            S = config[i, j]

            if j % 2 == 0:
                if i % 2 == 0:
                    nb = config[(i + 1) % N, j] + config[i, (j + 1) % N] + config[i, (j - 1) % N]
                else:
                    nb = config[(i - 1) % N, j] + config[i, (j + 1) % N] + config[i, (j - 1) % N]
            else:
                if i % 2 == 0:
                    nb = config[(i - 1) % N, j] + config[i, (j + 1) % N] + config[i, (j - 1) % N]
                else:
                    nb = config[(i + 1) % N, j] + config[i, (j + 1) % N] + config[i, (j - 1) % N]
            energy += -nb * S
    return energy / 2


def calcEnergy_3d(config, N):
    """
    Подсчёт энергии в заданной 3D решётке
    """
    energy = 0
    for a in range(N):
        for b in range(N):
            for c in range(N):
                S = config[a, b, c]

                if b % 2 == 0:
                    if a % 2 == 0:
                        nb = config[(a + 1) % N, b, c] + config[a, (b + 1) % N, c] + config[
                            a, (b - 1) % N, c] + \
                             config[a, b, (c + 1) % N] + config[a, b, (c - 1) % N]
                    else:
                        nb = config[(a - 1) % N, b, c] + config[a, (b + 1) % N, c] + config[
                            a, (b - 1) % N, c] + \
                             config[a, b, (c + 1) % N] + config[a, b, (c - 1) % N]
                else:
                    if a % 2 == 0:
                        nb = config[(a - 1) % N, b, c] + config[a, (b + 1) % N, c] + config[
                            a, (b - 1) % N, c] + \
                             config[a, b, (c + 1) % N] + config[a, b, (c - 1) % N]
                    else:
                        nb = config[(a + 1) % N, b, c] + config[a, (b + 1) % N, c] + config[
                            a, (b - 1) % N, c] + \
                             config[a, b, (c + 1) % N] + config[a, b, (c - 1) % N]

                energy += -nb * S

    return energy / 2.


def ising(calc, proc, N, d):
    """
    В данной функции происходит моделирование системы на заданных температурных точках,
    все температурные точки одинаково распределены между всеми логическими процессорами
    компьютера.
    """
    # следующие 4 строки нужны для приблизительного подсчёта времени, нужного для полного моделирования
    # системы
    flag = True
    if proc == 0:
        start = time.time()
        flag = False
    # следующая строка нужна для создания массивов для записи физических величин
    par = np.zeros(4 * calc).reshape((4, calc))
    for i in range(calc):
        # в строчке ниже происходит собственно моделирование и запись величины в точке температуры
        par[0][i], par[1][i], par[2][i], par[3][i] = sim_tt(N, (calc * proc + i), d)
        # в условии ниже происходит приблизительный расчёт нужного времени
        if flag == False:
            flag = True
            left = ((time.time() - start) * ((100 / (int((1 / calc) * 10000) / 100)) - 1) / 60)
            print(
                f"\n{left:.2f} minutes left for №{globals()[f'NN_{d}'].index(N) + 1}/{len(globals()[f'NN_{d}'])} test in {d}d.")
            print(f"сurrent time = {datetime.now().strftime('%H:%M:%S')}")
    # в цикле ниже происходит запись физических величин. Это необходимо для связи полученных данных
    # от каждого логического процессора, поскольку глобальные переменные не меняются при
    # параллельном программировании.
    for i in range(4):
        file = open(f"{'EMCX'[i]}_{proc}.txt", "w")
        file.write(str(par[i].tolist()))
        file.close()


def sim_tt(N, tt, d):
    """
    В данной функции происходит собственно моделирование случайно заданной системы и
    подсчёт полученных физических величин в конкретной температурной точке tt.
    """
    beta = 1.0 / globals()[f'T_{d}'][tt]  # обратная температура
    Ene = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)  # создаётся массив для записи энергии системы в разные моменты времени
    Mag = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)  # создаётся массив для записи намагниченности системы в разные моменты времени

    # следующие 6 строк нужны только для разделения моделирования двумерной системы от трёхмерной
    if d == 2:
        config = ((2 * np.random.randint(2, size=(N, N))) - 1)
    else:
        config = ((2 * np.random.randint(2, size=(N, N, N))) - 1)
    calcEnergy = f"calcEnergy_{d}d"
    mcmove = f"mcmove_{d}d"

    for i in range(eqSteps):  # МК шаги до прихода в равновесное состояние
        config = eval(mcmove)(config, beta, N)
    for i in range(mcSteps):  # МК шаги после прихода в установившееся состояние для подсчёта физических величин
        config = eval(mcmove)(config, beta, N)
        Ene[i] = eval(calcEnergy)(config, N)  # считаем энергию
        Mag[i] = np.sum(config, dtype=np.longdouble)  # считаем намагниченность
    # следующая строка считает среднюю энергию, среднюю намагниченность, теплоёмкость и магнитную восприимчивость
    E_mean, M_mean, C, X = np.mean(Ene), np.mean(Mag), beta ** 2 * np.std(Ene) ** 2, beta * np.std(Mag) ** 2
    return E_mean / N ** d, M_mean / N ** d, C / N ** d, X / N ** d


def processed(procs, calc, N, d):
    """
    Эта функция позволяет запустить мультипоточность моделирования.
    Мультипоточность достигается за счёт равномерного распределения температурных точек между
    всеми логическими процессорами компьютера. Это единственный способ использования принципа
    параллельного программирования, так как алгоритм Метрополиса относится к алгоритмам Марковский цепей,
    то есть моделирование каждого отдельного шага МК возможно только при известном предыдущем состоянии
    """
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=ising, args=(calc, proc, N, d))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


# запуск строки ниже запустит код при заданных параметрах
if __name__ == "__main__":
    # для подсчёта времени
    Start = time.time()
    # создание папок для создания массивов с физическими величинами для дальнейшего их анализа
    for d in D:
        dir_name = f'data_hex_{d}d_ising'
        output_filename = f'data_hex_{d}d_ising'
        basedir = os.path.abspath(os.getcwd())
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        # далее моделирование для каждого размера N каждой размерности d
        for N in globals()[f'NN_{d}']:
            processed(n_proc, calc, N, d)  # строка, запускающее всё то, что было выше
            # строчки ниже для объединения всех массивов от каждого потока в один единый
            E, M, C, X = [], [], [], []
            for i in range(n_proc):
                with open(f"E_{i}.txt", "r") as f:
                    E.append(eval(f.readline()))
                with open(f"M_{i}.txt", "r") as f:
                    M.append(eval(f.readline()))
                with open(f"C_{i}.txt", "r") as f:
                    C.append(eval(f.readline()))
                with open(f"X_{i}.txt", "r") as f:
                    X.append(eval(f.readline()))
                os.remove(f"E_{i}.txt")
                os.remove(f"M_{i}.txt")
                os.remove(f"C_{i}.txt")
                os.remove(f"X_{i}.txt")
            E = [a for b in E for a in b]
            M = np.array([a for b in M for a in b])
            C = [a for b in C for a in b]
            X = [a for b in X for a in b]

            # создание .zip архива с результатами каждого отдельного моделирования
            for i in range(5):
                file = open(f"{basedir}/{dir_name}/{'EMCXT'[i]}_{d}d_N={N}.txt", "w")
                file.write(str([E, M.tolist(), C, X, globals()[f'T_{d}'].tolist()][i]))
                file.close()
        shutil.make_archive(output_filename, 'zip', dir_name)
        shutil.rmtree(dir_name)
    # код ниже объединяет все полученные графики на одном
    for d in D:
        dir_name = f'data_hex_{d}d_ising'
        basedir = os.path.abspath(os.getcwd())
        zip_name = f'data_hex_{d}d_ising.zip'
        with zipfile.ZipFile(f"{zip_name}", 'r') as zip_ref:
            zip_ref.extractall(f"{basedir}/{dir_name}")
        for N in globals()[f"NN_{d}"]:
            name = f"{d}d_N={N}.txt"
            with open(f"{basedir}/{dir_name}/E_{name}", "r") as f:
                globals()[f"E_{d}d_N={N}"] = (eval(f.readline()))
            with open(f"{basedir}/{dir_name}/M_{name}", "r") as f:
                globals()[f"M_{d}d_N={N}"] = abs(np.array(eval(f.readline()))).tolist()
            with open(f"{basedir}/{dir_name}/C_{name}", "r") as f:
                globals()[f"C_{d}d_N={N}"] = (eval(f.readline()))
            with open(f"{basedir}/{dir_name}/X_{name}", "r") as f:
                globals()[f"X_{d}d_N={N}"] = (eval(f.readline()))
            with open(f"{basedir}/{dir_name}/T_{name}", "r") as f:
                globals()[f"T_{d}d_N={N}"] = (eval(f.readline()))
        f = plt.figure(figsize=(18, 10))
        titles = ['Энергия', 'Намагниченность', 'Теплоёмкость', 'Магн. восприимчивость']
        for j in range(4):
            title = titles[j]
            letter = 'EMCX'[j]
            labels = ['$E$', '$M$', '$C_v$', '$\chi$']
            ax = plt.subplot(2, 2, j + 1)
            for i in range(len(globals()[f"NN_{d}"])):
                N = globals()[f"NN_{d}"][i]
                plt.scatter(globals()[f"T_{d}d_N={N}"], globals()[f"{letter}_{d}d_N={N}"], s=8, label=f"N={N}")
            # crit = globals()[f"{letter}_{d}d_N={N}"].index(max(globals()[f"{letter}_{d}d_N={N}"]))
            ax.set_xlabel(r"$\frac{k_BT}{J}$", fontsize=15, fontweight="bold")
            label = str(labels[j])
            ax.set_ylabel(f"{label}", fontsize=15, fontweight="bold")
            ax.axis('tight')
            ax.set_title(title, fontsize=20, fontweight="bold")
            ax.grid('--', alpha=0.5)
            f.tight_layout(pad=3.0)
            ax.legend(loc='best')
        plt.savefig(f"plot_hex_ising_{d}d.png", bbox_inches='tight', dpi=400)
        plt.show()

    # вывод общего затраченного времени
    print(f'\ntotal time {((time.time() - Start) / 60):.2f} minutes')
