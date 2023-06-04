"""
Данный код позволяет смоделировать квадратную модель Поттса без внешнего поля.

Пользователь при желании может задать любые желаемые размеры решётки N×N (N - натуральное число),
любой температурный интервал T, а также желаемое количество температурных точек для создания графика nt,
округлённое вверх до числа, кратного количеству логических процессоров на компьютере.

По умолчанию установлены значения, при которых за достаточно короткое время можно
пронаблюдать графики, характерные для фазовых переходов второго рода.

В процессе работы происходит приблизительный подсчёт требуемого количества времени.

В коде могут присутствовать наработки, относящиеся к предполагаемой дальнейшей работе в этой теме.

При моделировании строго больше 6 случаев в одной размерности
не будут созданы изображения степенных аппроксимаций ветвей графиков.
"""

# Импортирование используемых библиотек
import math
import shutil
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib
import time
import multiprocessing
import os
from datetime import datetime
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
import zipfile

# Добавляем LaTeX для красивых подписей на графиках
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{palatino} \usepackage{textcomp}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']

"""
Переменные ниже задают параметры моделирования.
"""
n_proc = multiprocessing.cpu_count()
eqSteps = int(3 * 1e2)
mcSteps = int(1e3)
it = 48
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)
NN_2 = [5, 6, 7]
NN_3 = [15]
D = [2]
T_2 = np.linspace(0.5, 2.0, nt)
T_3 = np.linspace(3.7, 5.5, nt)
J = -1
q = 2


def initialise_2d(q, n):
    return (np.random.randint(q, size=(n, n)))


def initialise_3d(q, n):
    return (np.random.randint(q, size=(n, n, n)))


def mcmove_2d(config, beta, n, J, q):
    """
    Данная функция позволяет провести один шаг моделирования Монте-Карло
    для двумерной решётки методом Метрополиса.
    """
    for _ in range(n ** 2):
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        cur = config[i, j]
        new = np.random.randint(0, q)
        if new == cur:
            while new == cur:
                new = np.random.randint(0, q)
        sumstates = np.zeros(q)
        newsum = np.zeros(q)
        for __ in range(0, 3, 2):
            if config[(i + (__ - 1)) % n, j] == cur:
                sumstates[cur] += 1
            if config[(i + (__ - 1)) % n, j] == new:
                newsum[new] += 1
        for __ in range(0, 3, 2):
            if config[i, (j + (__ - 1)) % n] == cur:
                sumstates[cur] += 1
            if config[i, (j + (__ - 1)) % n] == new:
                newsum[new] += 1
        dE = (sum(newsum) - sum(sumstates)) * J * -1
        # ниже собственно работа метода Метрополиса
        if dE < 0 or rand() < np.exp(-dE * beta):
            config[i, j] = new
    return config


def mcmove_3d(config, beta, B, N, J, q):
    """
    Данная функция позволяет провести один шаг моделирования Монте-Карло
    для трёхмерной решётки методом Метрополиса.
    """
    for _ in range(N ** 3):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        k = np.random.randint(0, N)
        nb = J * ((B ** (((i + 1) % N) != (i + 1))) * config[(i + 1) % N, j, k]
                  + (B ** (((j + 1) % N) != (j + 1))) * config[i, (j + 1) % N, k]
                  + (B ** (((i - 1) % N) != (i - 1))) * config[(i - 1), j, k]
                  + (B ** (((j - 1) % N) != (j - 1))) * config[i, (j - 1), k]
                  + (B ** (((k + 1) % N) != (k + 1))) * config[i, j, (k + 1) % N]
                  + (B ** (((k - 1) % N) != (k - 1))) * config[i, j, (k - 1)])
        # nb - сумма изменений энергии в соседях
        dE = 2 * config[i, j, k] * nb
        # ниже собственно метод Метрополиса
        if dE < 0 or rand() < np.exp(-dE * beta):
            config[i, j, k] *= -1
    return config


def calcEnergy_2d(config, n, J, q):
    sumstates = np.zeros(q)
    for i in range(n):
        for j in range(n):
            for _ in range(0, 3, 2):
                if config[(i + (_ - 1)) % n, j] == config[i, j]:
                    sumstates[config[i, j]] += 1
            for _ in range(0, 3, 2):
                if config[i, (j + (_ - 1)) % n] == config[i, j]:
                    sumstates[config[i, j]] += 1
    return sum(sumstates)/4*J*(-1)


def calcEnergy_3d(config, bb, n, J, q):
    """
    Подсчёт энергии в заданной 3D решётке
    """
    energy = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                energy += J * -((bb ** (((i + 1) % n) != (i + 1))) * config[(i + 1) % n, j, k]
                                + (bb ** (((j + 1) % n) != (j + 1))) * config[i, (j + 1) % n, k]
                                + (bb ** (((i - 1) % n) != (i - 1))) * config[(i - 1) % n, j, k]
                                + (bb ** (((j - 1) % n) != (j - 1))) * config[i, (j - 1) % n, k]
                                + (bb ** (((k + 1) % n) != (k + 1))) * config[i, j, (k + 1) % n]
                                + (bb ** (((k - 1) % n) != (k - 1))) * config[i, j, (k - 1) % n]) * config[i, j, k]
    return energy / 6.

def calcMagn_2d(config, n):
    m = 0
    for i in range(n):
        for j in range(n):
            if config[i][j] == 0:
                m += 1
    return m

def potts(calc, proc, N, d, J, q):
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
        par[0][i], par[1][i], par[2][i], par[3][i] = sim_tt(N, (calc * proc + i), d, J, q)
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


def sim_tt(N, tt, d, J, q):
    """
    В данной функции происходит собственно моделирование случайно заданной системы и
    подсчёт полученных физических величин в конкретной температурной точке tt.
    """
    beta = abs(J) / globals()[f'T_{d}'][tt]  # обратная температура
    Ene = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)  # создаётся массив для записи энергии системы в разные моменты времени
    Mag = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)  # создаётся массив для записи намагниченности системы в разные моменты времени

    # следующие 6 строк нужны только для разделения моделирования двумерной системы от трёхмерной
    if d == 2:
        config = np.random.randint(q, size=(N, N))
    else:
        config = np.random.randint(q, size=(N, N, N))
    calcEnergy = f"calcEnergy_{d}d"
    mcmove = f"mcmove_{d}d"
    calcMagn = f"calcMagn_{d}d"

    for i in range(eqSteps):  # МК шаги до прихода в равновесное состояние
        config = eval(mcmove)(config, beta, N, J, q)
    for i in range(mcSteps):  # МК шаги после прихода в установившееся состояние для подсчёта физических величин
        config = eval(mcmove)(config, beta, N, J, q)
        Ene[i] = eval(calcEnergy)(config, N, J, q)  # считаем энергию
        Mag[i] = eval(calcMagn)(config, N) #np.sum(config, dtype=np.longdouble)  # считаем намагниченность
    # следующая строка считает среднюю энергию, среднюю намагниченность, теплоёмкость и магнитную восприимчивость
    E_mean, M_mean, C, X = np.mean(Ene), np.mean(Mag), beta ** 2 * np.std(Ene) ** 2, beta * np.std(Mag) ** 2
    if d == 2:
        states = np.zeros(q)
        for i in range(N):
            for j in range(N):
                states[config[i,j]] += 1
    return E_mean / N ** d, (q*(max(states)/(N**d)))/(q-1), C / N ** d, X / N ** d


def processed(procs, calc, N, d, J, q):
    """
    Эта функция позволяет запустить мультипоточность моделирования.
    Мультипоточность достигается за счёт равномерного распределения температурных точек между
    всеми логическими процессорами компьютера. Это единственный способ использования принципа
    параллельного программирования, так как алгоритм Метрополиса относится к алгоритмам Марковский цепей,
    то есть моделирование каждого отдельного шага МК возможно только при известном предыдущем состоянии
    """
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=potts, args=(calc, proc, N, d, J, q))
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
        dir_name = f'data_sq_{d}d_potts'
        output_filename = f'data_sq_{d}d_potts'
        basedir = os.path.abspath(os.getcwd())
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)

        # далее моделирование для каждого размера N каждой размерности d
        for N in globals()[f'NN_{d}']:
            processed(n_proc, calc, N, d, J, q)  # строка, запускающее всё то, что было выше
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
        dir_name = f'data_sq_{d}d_potts'
        basedir = os.path.abspath(os.getcwd())
        zip_name = f'data_sq_{d}d_potts.zip'
        with zipfile.ZipFile(f"{zip_name}", 'r') as zip_ref:
            zip_ref.extractall(f"{basedir}/{dir_name}")
        for N in globals()[f"NN_{d}"]:
            name = f"{d}d_N={N}.txt"
            with open(f"{basedir}\\{dir_name}\\E_{name}", "r") as f:
                globals()[f"E_{d}d_N={N}"] = (eval(f.readline()))
            with open(f"{basedir}\\{dir_name}\\M_{name}", "r") as f:
                globals()[f"M_{d}d_N={N}"] = abs(np.array(eval(f.readline()))).tolist()
            with open(f"{basedir}\\{dir_name}\\C_{name}", "r") as f:
                globals()[f"C_{d}d_N={N}"] = (eval(f.readline()))
            with open(f"{basedir}\\{dir_name}\\X_{name}", "r") as f:
                globals()[f"X_{d}d_N={N}"] = (eval(f.readline()))
            with open(f"{basedir}\\{dir_name}\\T_{name}", "r") as f:
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
            ax.set_xlabel(r"$\frac{k_BT}{|\;J\;|}$", fontsize=15, fontweight="bold")
            crit = abs(J) / (math.log(1+q**(1/2)))
            plt.plot([crit, crit], [max(globals()[f"{letter}_{d}d_N={N}"]), min(globals()[f"{letter}_{d}d_N={N}"])],
                     c='r', alpha=0.5, label=f"$T_c=${crit:.3f} (т.)")
            label = str(labels[j])
            ax.set_ylabel(f"{label}", fontsize=15, fontweight="bold")
            ax.axis('tight')
            ax.set_title(title, fontsize=20, fontweight="bold")
            ax.grid('--', alpha=0.5)
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            f.tight_layout(pad=3.0)
            ax.legend(loc='best')
        plt.savefig(f"plot_sq_potts_{d}d.png", bbox_inches='tight', dpi=400)
        plt.show()

    # вывод общего затраченного времени
    print(f'\ntotal time {((time.time() - Start) / 60):.2f} minutes')


    # далее подсчёт альфы

    def func_powerlaw_2d(T, k, alpha, Tc=2.269):
        return k * np.power(np.abs((T - Tc) / Tc), alpha)


    def func_powerlaw_3d(T, k, alpha, Tc=4.5):
        return k * np.abs((T - Tc) / Tc) ** alpha


    def powerlaw_2d(x, A, v, b=2.269):
        return A * np.power(x, v) + b


    def powerlaw_3d(x, A, v, b=4.5):
        return A * np.power(x, v) + b


    for d in D:
        dir_name = f'data_sq_{d}d_potts'
        basedir = os.path.abspath(os.getcwd())
        zip_name = f'data_sq_{d}d_potts.zip'
        with zipfile.ZipFile(f"{zip_name}", 'r') as zip_ref:
            zip_ref.extractall(f"{basedir}/{dir_name}")
        Chis = []
        for N in globals()[f"NN_{d}"]:
            name = f"{d}d_N={N}.txt"
            with open(f"{basedir}\\{dir_name}\\E_{name}", "r") as f:
                globals()[f"E_{d}d_N={N}"] = (eval(f.readline()))
            with open(f"{basedir}\\{dir_name}\\M_{name}", "r") as f:
                globals()[f"M_{d}d_N={N}"] = abs(np.array(eval(f.readline()))).tolist()
            with open(f"{basedir}\\{dir_name}\\C_{name}", "r") as f:
                globals()[f"C_{d}d_N={N}"] = (eval(f.readline()))
            with open(f"{basedir}\\{dir_name}\\X_{name}", "r") as f:
                globals()[f"X_{d}d_N={N}"] = (eval(f.readline()))
            with open(f"{basedir}\\{dir_name}\\T_{name}", "r") as f:
                globals()[f"T_{d}d_N={N}"] = (eval(f.readline()))
            Chis.append(globals()[f"C_{d}d_N={N}"])
        plt.figure(figsize=(16, 15))
        n = len(globals()[f"NN_{d}"])
        Ks = np.zeros(n)
        alphas = np.zeros(n)
        sigma_alpha = np.zeros(n)
        TCs = np.zeros(n)
        sigma_TCs = np.zeros(n)
        show = True
        for i, Chi in enumerate(Chis):
            C = Chi
            N = globals()[f"NN_{d}"][i]
            T = np.linspace(globals()[f"T_{d}d_N={N}"][0], globals()[f"T_{d}d_N={N}"][-1], len(Chi))
            if len(Chis) > 6:
                show = False
                Tmax = np.argmax(Chi)
                Tmax = C.index(Chi[Tmax])
                Chi = C
                T = np.linspace(globals()[f"T_{d}d_N={N}"][0], globals()[f"T_{d}d_N={N}"][-1], len(Chi))
                sol, cov = curve_fit(eval(f"func_powerlaw_{d}d"), T[Tmax:], Chi[Tmax:], maxfev=int(1e9))
                Ks[i] = sol[0]
                alphas[i] = sol[1]
                TCs[i] = sol[2]
                sigma_alpha[i] = cov[1, 1]
                sigma_TCs[i] = cov[2, 2]
            else:
                ax = plt.subplot(3, 2, i + 1)
                Tmax = np.argmax(Chi)
                Tmax = C.index(Chi[Tmax])
                Chi = C
                T = np.linspace(globals()[f"T_{d}d_N={N}"][0], globals()[f"T_{d}d_N={N}"][-1], len(Chi))
                plt.scatter(T[Tmax:], Chi[Tmax:], s=0.6)
                sol, cov = curve_fit(eval(f"func_powerlaw_{d}d"), T[Tmax:], Chi[Tmax:], maxfev=int(1e9))
                Ks[i] = sol[0]
                alphas[i] = sol[1]
                TCs[i] = sol[2]
                sigma_alpha[i] = cov[1, 1]
                sigma_TCs[i] = cov[2, 2]
                ax.plot(T[Tmax:], eval(f"func_powerlaw_{d}d")(T[Tmax:], Ks[i], alphas[i], TCs[i]), 'orange',
                        label='fit')
                ax.scatter(T[Tmax:], Chi[Tmax:], label=f'N={N}')
                ax.text(0.8, 0.5, r'$\alpha$' + '= {}\n$T_c$ = {}'.format('%.3f' % (-1 * alphas[i]), '%.3f' % TCs[i]),
                        transform=ax.transAxes,
                        bbox=dict(alpha=0.7), fontsize=10)
                ax.set_xlabel('Температура', fontsize=10)
                ax.set_ylabel('Теплоёмкость', fontsize=10)
                ax.set_title(f"Аппроксимация N={N}, d={d}", fontsize=10, fontweight="bold")
                plt.legend(loc='best')
                plt.subplots_adjust(hspace=0.45)
                plt.grid()
        if show:
            plt.show()

        n = [(1 / n) for n in globals()[f"NN_{d}"]]
        sol, cov = curve_fit(eval(f"powerlaw_{d}d"), n, TCs, maxfev=int(1e9))
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.errorbar(n, TCs, yerr=sigma_TCs, fmt='o', capsize=4)
        plt.xlabel(r'$\frac{1}{N}$', fontsize=16)
        plt.ylabel('$T$', fontsize=16)
        plt.title('Критическая температура $T_c$', fontsize=20, fontweight="bold")
        crit = round(crit, 3)
        plt.axhline(y=crit, linestyle='--', color='red', label='$T_{C, t}$')
        plt.grid()
        plt.legend(fontsize=18, loc='best')
        plt.subplot(1, 2, 2)
        plt.errorbar(n, -alphas, yerr=sigma_alpha, fmt='o', capsize=4)
        plt.xlabel(r'$\frac{1}{N}$', fontsize=16)
        plt.title(r'Критический показатель $\alpha$', fontsize=20, fontweight="bold")
        # if d == 2:
        #     th = 0
        # else:
        #     th = 1.25 / 10
        # plt.axhline(y=th, linestyle='--', color=('red'), label=r'$\alpha_{theory}$')
        plt.grid()
        # plt.legend(fontsize=18, loc='best')
        plt.show()
