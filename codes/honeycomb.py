"""
Данный код позволяет смоделировать шестиугольную модель Изинга без внешнего поля.

Пользователь при желании может задать любые желаемые размеры решётки N×N (N - натуральное число, означающее
количество шестиугольников в одном ряду), любой температурный интервал T,
а также желаемое количество температурных точек для создания графика nt,
округлённое вверх до числа, кратного количеству логических процессоров на компьютере.

По умолчанию установлены значения, при которых за достаточно короткое время можно
пронаблюдать графики, характерные для фазовых переходов второго рода.

Код также выводит численные значения полученных физических величин для конкретного N и конкретного d для
дальнейшего более внимательного анализа на случай неправильно выбранной точки вершины.

В процессе работы происходит приблизительный подсчёт требуемого количества времени.

В коде могут присутствовать наработки, относящиеся к предполагаемой дальнейшей работе в этой теме.

При моделировании строго больше 6 случаев в одной размерности
не будут созданы изображения степенных аппроксимаций ветвей графиков.
"""

# Импортирование используемых библиотек
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
n_proc = multiprocessing.cpu_count()  # подсчёт логических операторов на компьютере
eqSteps = int(3 * 1e2)  # количество МК шагов для прихода в равновесное состояние (рекомендую 300)
mcSteps = int(1e3)  # количество МК шагов для подсчёта физических величин (рекомендую 1000)
it = 20  # количество желаемых температурных точек (рекомендую 100)
calc = it // n_proc + ((it // n_proc) != (it / n_proc))  # количество температурных точек на один логический процессор
nt = int(calc * n_proc)  # реальное количество используемых температурных точек
NN_2 = [2]  # размеры желанных 2D решёток (рекомендую [15, 20])
NN_3 = [15]  # размеры желанных 3D решёток (не рекомендую)
D = [2]  # желаемые размерности (2 или 3) (рекомендую [2])
T_2 = np.linspace(0.1, 5.1, nt)  # желаемый температурный интервал для 2D (рекомендую (1.5, 2.5, nt))
T_3 = np.linspace(3.5, 5.5, nt)  # желаемый температурный интервал для 3D (рекомендую (3.5, 5.5, nt))


def initialise_2d(n):
    Nx = (2 * n) + 2
    Ny = (2 * n) + 1
    config = (((2 * np.random.randint(2, size=(Ny, Nx))) - 1))
    for y in range(1, Ny + 1, 2):
        for x in range(0, Nx, 4):
            config[-y][x] = 0
    for y in range(1, Ny + 1, 2):
        for x in range(3, Nx, 4):
            config[-y][x] = 0
    for y in range(3, Ny + 1, 2):
        for x in range(0, Nx, 4):
            if n % 2 == 0 and (x + 2) >= Nx:
                break
            else:
                config[-y + 1][x + 1] = 0
                config[-y + 1][x + 2] = 0
    if n % 2 == 0:
        for y in range(1, Ny, 2):
            config[y][-1] = 0
    return config


def initialise_3d(n):
    config = []
    for i in range(n):
        config.append(initialise_2d(n))
    return np.array(config)


def mcmove_2d(config, beta, n):
    """
    Monte Carlo move using Metropolis algorithm
    """
    Nx = (2 * n) + 2
    Ny = (2 * n) + 1
    _ = 0
    while _ != n ** 2:
        a = np.random.randint(n)
        b = np.random.randint(n)
        s = config[a][b]
        if s == 0:
            pass
        else:
            _ += 1

            if a % 2 == 0:  # строки начинающиеся 0
                if config[a][b - 1] == 0:
                    if a != 0 and a != Ny - 1:
                        nb = config[a - 1][b - 1] + config[a + 1][b - 1] + config[a][b + 1]
                    elif a == Ny - 1:
                        nb = config[a - 1][b - 1] + config[1][b - 1] + config[a][b + 1]
                    elif a == 0:
                        nb = config[-2][b - 1] + config[a + 1][b - 1] + config[a][b + 1]
                else:
                    if a != 0 and a != Ny - 1:
                        nb = config[a - 1][b + 1] + config[a + 1][b + 1] + config[a][b - 1]
                    elif a == 0:
                        nb = config[-2][b + 1] + config[a + 1][b + 1] + config[a][b - 1]
                    elif a == Ny - 1:
                        nb = config[a - 1][b + 1] + config[1][b + 1] + config[a][b - 1]
            else:  # строки начинающиеся 1
                if b != 0 and b != Nx - 1:
                    if config[a][b - 1] == 0:
                        nb = config[a][b + 1] + config[a - 1][b - 1] + config[a + 1][b - 1]
                    else:
                        nb = config[a][b - 1] + config[a - 1][b + 1] + config[a + 1][b + 1]
                elif b == 0:
                    nb = config[a - 1][b + 1] + config[a + 1][b + 1] + config[a][-1 - (n % 2 == 0)]
                elif b == Nx - 1:
                    nb = config[a - 1][b - 1] + config[a + 1][b - 1] + config[a][0]
            cost = 2 * s * nb
            if cost < 0 or rand() < np.exp(-cost * beta):
                config[a, b] = -s
    return config


def calcEnergy_2d(config, n):
    """
    Energy of a given configuration
    """
    energy = 0
    Nx = (2 * n) + 2
    Ny = (2 * n) + 1
    for a in range(len(config)):
        for b in range(len(config)):
            if config[a, b] == 0:
                pass
            else:
                if a % 2 == 0:
                    if config[a][b - 1] == 0:
                        if a != 0 and a != Ny - 1:
                            nb = config[a - 1][b - 1] + config[a + 1][b - 1] + config[a][b + 1]
                        elif a == Ny - 1:
                            nb = config[a - 1][b - 1] + config[1][b - 1] + config[a][b + 1]
                        elif a == 0:
                            nb = config[-2][b - 1] + config[a + 1][b - 1] + config[a][b + 1]
                    else:
                        if a != 0 and a != Ny - 1:
                            nb = config[a - 1][b + 1] + config[a + 1][b + 1] + config[a][b - 1]
                        elif a == 0:
                            nb = config[-2][b + 1] + config[a + 1][b + 1] + config[a][b - 1]
                        elif a == Ny - 1:
                            nb = config[a - 1][b + 1] + config[1][b + 1] + config[a][b - 1]
                else:
                    if b != 0 and b != Nx - 1:
                        if config[a][b - 1] == 0:
                            nb = config[a][b + 1] + config[a - 1][b - 1] + config[a + 1][b - 1]
                        else:
                            nb = config[a][b - 1] + config[a - 1][b + 1] + config[a + 1][b + 1]
                    elif b == 0:
                        nb = config[a - 1][b + 1] + config[a + 1][b + 1] + config[a][-1 - (n % 2 == 0)]
                    elif b == Nx - 1:
                        nb = config[a - 1][b - 1] + config[a + 1][b - 1] + config[a][0]
                energy += -nb * config[a, b]
    return energy / 3.


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
        parametrs = sim_tt(N, (calc * proc + i), d)
        for j in range(4):
            par[j][i] = parametrs[j]
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
    beta = abs(J) / globals()[f'T_{d}'][tt]  # обратная температура
    Ene = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)  # создаётся массив для записи энергии системы в разные моменты времени
    Mag = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)  # создаётся массив для записи намагниченности системы в разные моменты времени

    # следующие 6 строк нужны только для разделения моделирования двумерной системы от трёхмерной
    if d == 2:
        config = initialise_2d(N)
    else:
        config = initialise_3d(N)
    calcEnergy = f"calcEnergy_{d}d"
    mcmove = f"mcmove_{d}d"

    for i in range(eqSteps):  # МК шаги до прихода в равновесное состояние
        config = eval(mcmove)(config, beta, N)
    for i in range(mcSteps):  # МК шаги после прихода в установившееся состояние для подсчёта физических величин
        config = eval(mcmove)(config, beta, N)
        Ene[i] = eval(calcEnergy)(config, N)  # считаем энергию
        Mag[i] = np.sum(config, dtype=np.longdouble)  # считаем намагниченность
    # следующая строка считает среднюю энергию, среднюю намагниченность, теплоёмкость и магнитную восприимчивость
    E_mean, M_mean, C, X = np.mean(Ene), np.mean(Mag), \
                           beta ** 2 * np.std(Ene, dtype=np.float64) ** 2, \
                           beta * np.std(Mag, dtype=np.float64) ** 2
    if d == 2:
        summ = sum(sum(abs(config)))
    else:
        summ = sum(sum(sum(config)))
    return [E_mean / summ, M_mean / summ, C / summ, X / summ]


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
            crit = globals()[f"{letter}_{d}d_N={N}"].index(max(globals()[f"{letter}_{d}d_N={N}"]))
            ax.set_xlabel(r"$\frac{k_BT}{|\;J\;|}$", fontsize=15, fontweight="bold")
            label = str(labels[j])
            ax.set_ylabel(f"{label}", fontsize=15, fontweight="bold")
            ax.axis('tight')
            ax.set_title(title, fontsize=20, fontweight="bold")
            ax.grid('--', alpha=0.5)
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            f.tight_layout(pad=3.0)
            ax.legend(loc='best')
        plt.savefig(f"plot_hex_ising_{d}d.png", bbox_inches='tight', dpi=400)
        plt.show()

    # вывод общего затраченного времени
    print(f'\ntotal time {((time.time() - Start) / 60):.2f} minutes')


    # далее первичный (без хорошего учёта вершины) подсчёт альфы

    def func_powerlaw_2d(T, k, alpha, Tc=2.269):
        return k * np.power(np.abs((T - Tc) / Tc), alpha)


    def func_powerlaw_3d(T, k, alpha, Tc=4.5):
        return k * np.abs((T - Tc) / Tc) ** alpha


    def powerlaw_2d(x, A, v, b=2.269):
        return A * np.power(x, v) + b


    def powerlaw_3d(x, A, v, b=4.5):
        return A * np.power(x, v) + b


    for d in D:
        dir_name = f'data_hex_{d}d_ising'
        basedir = os.path.abspath(os.getcwd())
        zip_name = f'data_hex_{d}d_ising.zip'
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
        sol, cov = curve_fit(eval(f"powerlaw_{d}d"), n, TCs, maxfev=int(1e6))
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.errorbar(n, TCs, yerr=sigma_TCs, fmt='o', capsize=4)
        plt.xlabel(r'$\frac{1}{N}$', fontsize=16)
        plt.ylabel('$T$', fontsize=16)
        plt.title('Критическая температура $T_c$', fontsize=20, fontweight="bold")
        T_cr = np.average(TCs)
        plt.axhline(y=T_cr, linestyle='--', color=('red'), label='$T_{C}, av$')
        plt.grid()
        plt.legend(fontsize=18, loc='best')
        plt.subplot(1, 2, 2)
        plt.errorbar(n, -alphas, yerr=sigma_alpha, fmt='o', capsize=4)
        plt.xlabel(r'$\frac{1}{N}$', fontsize=16)
        plt.title(r'Критический показатель $\alpha$', fontsize=20, fontweight="bold")
        if d == 2:
            th = 0
        else:
            th = 1.25 / 10
        plt.axhline(y=th, linestyle='--', color=('red'), label=r'$\alpha_{theory}$')
        plt.grid()
        plt.legend(fontsize=18, loc='best')
        plt.show()
