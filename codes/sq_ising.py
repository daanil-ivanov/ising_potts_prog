import shutil
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import time
import multiprocessing
import os
from datetime import datetime
from matplotlib.ticker import MultipleLocator
plt.rcParams.update({"font.family": "serif", "font.cursive": ["Comic Neue", "Comic Sans MS"], })
n_proc = multiprocessing.cpu_count()
eqSteps = int(7e2)
mcSteps = int(2e3)
it = 200
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)
NN_2 = [10, 20, 25, 30, 35]
NN_3 = [10, 15, 20, 25]
D = [2, 3]
T_2 = np.linspace(2.2, 2.35, nt)
T_3 = np.linspace(4.42, 4.57, nt)
J = 1


def mcmove_2d(config, beta, N, J):
    for _ in range(N ** 2):
        i, j = np.random.randint(0, N), np.random.randint(0, N)
        nb = J * (config[(i + 1) % N, j] + config[i, (j + 1) % N] + config[(i - 1), j] + config[i, (j - 1)])
        # nb - сумма изменений энергии в соседях
        dE = 2 * config[i, j] * nb
        # ниже собственно работа метода Метрополиса
        if dE < 0 or rand() < np.exp(-dE * beta):
            config[i, j] *= -1
    return config


def mcmove_3d(config, beta, N, J):
    for _ in range(N ** 3):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        k = np.random.randint(0, N)
        nb = J * (config[(i + 1) % N, j, k]
                  + config[i, (j + 1) % N, k]
                  + config[(i - 1), j, k]
                  + config[i, (j - 1), k]
                  + config[i, j, (k + 1) % N]
                  + config[i, j, (k - 1)])
        # nb - сумма изменений энергии в соседях
        dE = 2 * config[i, j, k] * nb
        # ниже собственно метод Метрополиса
        if dE < 0 or rand() < np.exp(-dE * beta):
            config[i, j, k] *= -1
    return config


def calcEnergy_2d(config, n, J):
    energy = 0
    for i in range(n):
        for j in range(n):
            energy += -1 * J * (config[(i + 1) % n, j] + config[i, (j + 1) % n] + config[(i - 1) % n, j] + config[
                i, (j - 1) % n]) * config[i, j]
    return energy / 2.


def calcEnergy_3d(config, n, J):
    energy = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                energy += J * -(
                        config[(i + 1) % n, j, k] + config[i, (j + 1) % n, k] + config[(i - 1) % n, j, k] + config[
                    i, (j - 1) % n, k] + config[i, j, (k + 1) % n] + config[i, j, (k - 1) % n]) * config[i, j, k]
    return energy / 2.


def ising(calc, proc, N, d, J):
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
        par[0][i], par[1][i], par[2][i], par[3][i] = sim_tt(N, (calc * proc + i), d, J)
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


def sim_tt(N, tt, d, J):
    beta = abs(J) / globals()[f'T_{d}'][tt]  # обратная температура
    Ene = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)  # создаётся массив для записи энергии системы в разные моменты времени
    Mag = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)  # создаётся массив для записи намагниченности системы в разные моменты времени

    # следующие 6 строк нужны только для разделения моделирования двумерной системы от трёхмерной
    if d == 2:
        config = (((2 * np.random.randint(2, size=(N, N))) - 1))
    else:
        config = (((2 * np.random.randint(2, size=(N, N, N))) - 1))
    calcEnergy = f"calcEnergy_{d}d"
    mcmove = f"mcmove_{d}d"

    for i in range(eqSteps):  # МК шаги до прихода в равновесное состояние
        config = eval(mcmove)(config, beta, N, J)
    for i in range(mcSteps):  # МК шаги после прихода в установившееся состояние для подсчёта физических величин
        config = eval(mcmove)(config, beta, N, J)
        Ene[i] = eval(calcEnergy)(config, N, J)  # считаем энергию
        Mag[i] = np.sum(config, dtype=np.longdouble)  # считаем намагниченность
    # следующая строка считает среднюю энергию, среднюю намагниченность, теплоёмкость и магнитную восприимчивость
    E_mean, M_mean, C, X = np.mean(Ene), np.mean(Mag), beta ** 2 * np.std(Ene) ** 2, beta * np.std(Mag) ** 2
    return E_mean / N ** d, M_mean / N ** d, C / N ** d, X / N ** d


def processed(procs, calc, N, d, J):
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=ising, args=(calc, proc, N, d, J))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    # Start = time.time()
    # for d in D:
    #     dir_name = f'data_{d}d_ising'
    #     output_filename = f'data_{d}d_ising'
    #     basedir = os.path.abspath(os.getcwd())
    #     if os.path.exists(dir_name):
    #         shutil.rmtree(dir_name)
    #     os.mkdir(dir_name)
    #     for N in globals()[f'NN_{d}']:
    #         processed(n_proc, calc, N, d, J)
    #         E, M, C, X = [], [], [], []
    #         for i in range(n_proc):
    #             with open(f"E_{i}.txt", "r") as f:
    #                 E.append(eval(f.readline()))
    #             with open(f"M_{i}.txt", "r") as f:
    #                 M.append(eval(f.readline()))
    #             with open(f"C_{i}.txt", "r") as f:
    #                 C.append(eval(f.readline()))
    #             with open(f"X_{i}.txt", "r") as f:
    #                 X.append(eval(f.readline()))
    #             os.remove(f"E_{i}.txt")
    #             os.remove(f"M_{i}.txt")
    #             os.remove(f"C_{i}.txt")
    #             os.remove(f"X_{i}.txt")
    #         E = [a for b in E for a in b]
    #         M = np.array([a for b in M for a in b])
    #         C = [a for b in C for a in b]
    #         X = [a for b in X for a in b]
    #         for i in range(5):
    #             file = open(f"{basedir}/{dir_name}/{'EMCXT'[i]}_{d}d_N={N}.txt", "w")
    #             file.write(str([E, M.tolist(), C, X, globals()[f'T_{d}'].tolist()][i]))
    #             file.close()
    #     shutil.make_archive(output_filename, 'zip', dir_name)
    #     shutil.rmtree(dir_name)
    for d in D:
        dir_name = f'data_{d}d_ising'
        basedir = os.path.abspath(os.getcwd())
        zip_name = f'data_{d}d_ising.zip'
        # with zipfile.ZipFile(f"{zip_name}", 'r') as zip_ref:
        #     zip_ref.extractall(f"{basedir}/{dir_name}")
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
        f.suptitle(f"Квадратная модель Изинга, $d$={d}", fontsize=25, fontweight="bold")
        titles = ['Энергия', 'Намагниченность', 'Теплоёмкость', 'Магн. восприимчивость']
        for j in range(4):
            title = titles[j]
            letter = 'EMCX'[j]
            labels = ['$E$', '$M$', '$C_v$', '$\chi$']
            ax = plt.subplot(2, 2, j + 1)
            for i in range(len(globals()[f"NN_{d}"])):
                N = globals()[f"NN_{d}"][i]
                plt.scatter(globals()[f"T_{d}d_N={N}"], globals()[f"{letter}_{d}d_N={N}"], s=3, label=f"N={N}")
            if d == 2:
                crit = 2.269
            if d == 3:
                crit = 4.511
            plt.axvline(x=crit, c='r', alpha=0.5, label=f"$T_c=${crit}")
            ax.set_xlabel(r"$\frac{k_BT}{|\;J\;|}$", fontsize=15, fontweight="bold")
            label = str(labels[j])
            ax.set_ylabel(f"{label}", fontsize=15, fontweight="bold")
            ax.axis('tight')
            ax.set_title(title, fontsize=20, fontweight="bold")
            ax.grid('--', alpha=0.5)
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            f.tight_layout(pad=3.0)
            ax.legend(loc='best', fontsize='12.5')
        plt.savefig(f"plot_ising_{d}d.png", bbox_inches='tight', dpi=300)
        plt.show()
    # print(f'\ntotal time {((time.time() - Start) / 60):.2f} minutes')