import math
import shutil
import numpy as np
from numpy.random import rand
import matplotlib
import matplotlib.pyplot as plt
import time
import multiprocessing
import os
from datetime import datetime
import zipfile
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{palatino} \usepackage{textcomp}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']

n_proc = multiprocessing.cpu_count()
eqSteps = int(5e2)
mcSteps = int(1e3)
it = 300
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)
NN_2 = [10, 20, 25, 30, 35]
NN_3 = [10, 20, 25, 30]
D = [2, 3]
Q = [2, 3]
T_2 = np.linspace(1.1, 1.2, nt)
T_3 = np.linspace(2.1, 2.25, nt)
J = 1
h = 0

def mcmove_2d(config, beta, n, J, q, h):
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
        if h != 0:
            dE += - h * calcMagn_2d(config, n, q)
        if dE < 0 or rand() < np.exp(-dE * beta):
            config[i, j] = new
    return config


def mcmove_3d(config, beta, n, J, q, h):
    for _ in range(n ** 3):
        i, j, k = np.random.randint(0, n), np.random.randint(0, n), np.random.randint(0, n)
        cur = config[i, j, k]
        new = np.random.randint(0, q)
        if new == cur:
            while new == cur:
                new = np.random.randint(0, q)
        sumstates = np.zeros(q)
        newsum = np.zeros(q)
        for __ in range(0, 3, 2):
            if config[(i + (__ - 1)) % n, j, k] == cur:
                sumstates[cur] += 1
            if config[(i + (__ - 1)) % n, j, k] == new:
                newsum[new] += 1
        for __ in range(0, 3, 2):
            if config[i, (j + (__ - 1)) % n, k] == cur:
                sumstates[cur] += 1
            if config[i, (j + (__ - 1)) % n, k] == new:
                newsum[new] += 1
        for __ in range(0, 3, 2):
            if config[i, j, (k + (__ - 1)) % n] == cur:
                sumstates[cur] += 1
            if config[i, j, (k + (__ - 1)) % n] == new:
                newsum[new] += 1
        dE = (sum(newsum) - sum(sumstates)) * J * -1
        if h != 0:
            dE += - h * calcMagn_3d(config, n, q)
        if dE < 0 or rand() < np.exp(-dE * beta):
            config[i, j, k] = new
    return config


def calcEnergy_2d(config, n, J, q, h):
    sumstates = np.zeros(q)
    for i in range(n):
        for j in range(n):
            for _ in range(0, 3, 2):
                if config[(i + (_ - 1)) % n, j] == config[i, j]:
                    sumstates[config[i, j]] += 1
            for _ in range(0, 3, 2):
                if config[i, (j + (_ - 1)) % n] == config[i, j]:
                    sumstates[config[i, j]] += 1
    energy = sum(sumstates) / 2 * J * (-1)
    if h != 0:
        energy += (calcMagn_2d(config, n, q) * h)
    return energy


def calcEnergy_3d(config, n, J, q, h):
    sumstates = np.zeros(q)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for _ in range(0, 3, 2):
                    if config[(i + (_ - 1)) % n, j, k] == config[i, j, k]:
                        sumstates[config[i, j, k]] += 1
                for _ in range(0, 3, 2):
                    if config[i, (j + (_ - 1)) % n, k] == config[i, j, k]:
                        sumstates[config[i, j, k]] += 1
                for _ in range(0, 3, 2):
                    if config[i, j, (k + (_ - 1)) % n] == config[i, j, k]:
                        sumstates[config[i, j, k]] += 1
    energy = sum(sumstates) / 2 * J * (-1)
    if h != 0:
        energy += (calcMagn_3d(config, n, q) * h)
    return energy


def calcMagn_2d(config, n, q):
    states = np.zeros(q)
    for i in range(n):
        for j in range(n):
            states[config[i, j]] += 1 / n ** 2
    sq_sum = 0
    for _ in range(q):
        sq_sum += states[_] ** 2
    m = q / (q - 1) * (sq_sum - 1 / q)
    return m * (n ** 2)


def calcMagn_3d(config, n, q):
    states = np.zeros(q)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                states[config[i, j, k]] += 1 / n ** 3
    sq_sum = 0
    for _ in range(q):
        sq_sum += states[_] ** 2
    m = q / (q - 1) * (sq_sum - 1 / q)
    return m * (n ** 3)


def potts(calc, proc, N, d, J, q, h):
    flag = True
    if proc == 0:
        start = time.time()
        flag = False
    par = np.zeros(4 * calc).reshape((4, calc))
    for i in range(calc):
        par[0][i], par[1][i], par[2][i], par[3][i] = sim_tt(N, (calc * proc + i), d, J, q, h)
        if flag == False:
            flag = True
            left = ((time.time() - start) * ((100 / (int((1 / calc) * 10000) / 100)) - 1) / 60)
            print(
                f"\n{left:.2f} minutes left for №{globals()[f'NN_{d}'].index(N) + 1}/{len(globals()[f'NN_{d}'])} test in {d}d q={q} N={N}")
            print(f"сurrent time = {datetime.now().strftime('%H:%M:%S')}")
    for i in range(4):
        file = open(f"{'EMCX'[i]}_{proc}.txt", "w")
        file.write(str(par[i].tolist()))
        file.close()


def sim_tt(N, tt, d, J, q, h):
    beta = abs(J) / globals()[f'T_{d}'][tt]
    Ene = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)
    Mag = np.array(np.zeros(mcSteps),
                   dtype=np.longdouble)
    if d == 2:
        config = np.random.randint(q, size=(N, N))
    else:
        config = np.random.randint(q, size=(N, N, N))
    calcEnergy = f"calcEnergy_{d}d"
    mcmove = f"mcmove_{d}d"
    calcMagn = f"calcMagn_{d}d"

    for i in range(eqSteps):
        config = eval(mcmove)(config, beta, N, J, q, h)
    for i in range(mcSteps):
        config = eval(mcmove)(config, beta, N, J, q, h)
        Ene[i] = eval(calcEnergy)(config, N, J, q, h)
        Mag[i] = (eval(calcMagn)(config, N, q))
    C, X = beta ** 2 * np.std(Ene) ** 2, beta * np.std(Mag) ** 2
    return np.mean(Ene) / N ** d, np.mean(Mag) / (N ** d), C / N ** d, X / N ** d


def processed(procs, calc, N, d, J, q, h):
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=potts, args=(calc, proc, N, d, J, q, h))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    Start = time.time()
    for q in Q:
        for d in D:
            if d == 2 or (q == 2 and d == 3):
                # dir_name = f'data_sq_{d}d_q={q}_potts'
                # output_filename = f'data_sq_{d}d_q={q}_potts'
                # basedir = os.path.abspath(os.getcwd())
                # if os.path.exists(dir_name):
                #     shutil.rmtree(dir_name)
                # os.mkdir(dir_name)
                # for N in globals()[f'NN_{d}']:
                #     processed(n_proc, calc, N, d, J, q, h)
                #     E, M, C, X = [], [], [], []
                #     for i in range(n_proc):
                #         with open(f"E_{i}.txt", "r") as f:
                #             E.append(eval(f.readline()))
                #         with open(f"M_{i}.txt", "r") as f:
                #             M.append(eval(f.readline()))
                #         with open(f"C_{i}.txt", "r") as f:
                #             C.append(eval(f.readline()))
                #         with open(f"X_{i}.txt", "r") as f:
                #             X.append(eval(f.readline()))
                #         os.remove(f"E_{i}.txt")
                #         os.remove(f"M_{i}.txt")
                #         os.remove(f"C_{i}.txt")
                #         os.remove(f"X_{i}.txt")
                #     E = [a for b in E for a in b]
                #     M = np.array([a for b in M for a in b])
                #     C = [a for b in C for a in b]
                #     X = [a for b in X for a in b]
                #     for i in range(5):
                #         file = open(f"{basedir}/{dir_name}/{'EMCXT'[i]}_{d}d_q={q}_N={N}.txt", "w")
                #         file.write(str([E, M.tolist(), C, X, globals()[f'T_{d}'].tolist()][i]))
                #         file.close()
                # shutil.make_archive(output_filename, 'zip', dir_name)
                # shutil.rmtree(dir_name)
                print(f'\ntotal time {((time.time() - Start) / 60):.2f} minutes')
                dir_name = f'data_sq_{d}d_q={q}_potts'
                basedir = os.path.abspath(os.getcwd())
                zip_name = f'data_sq_{d}d_q={q}_potts.zip'
                # with zipfile.ZipFile(f"{zip_name}", 'r') as zip_ref:
                #     zip_ref.extractall(f"{basedir}/{dir_name}")
                for N in globals()[f"NN_{d}"]:
                    name = f"{d}d_q={q}_N={N}.txt"
                    with open(f"{basedir}/{dir_name}/E_{name}", "r") as f:
                        globals()[f"E_{d}d_q={q}_N={N}"] = (eval(f.readline()))
                    with open(f"{basedir}/{dir_name}/M_{name}", "r") as f:
                        globals()[f"M_{d}d_q={q}_N={N}"] = abs(np.array(eval(f.readline()))).tolist()
                    with open(f"{basedir}/{dir_name}/C_{name}", "r") as f:
                        globals()[f"C_{d}d_q={q}_N={N}"] = (eval(f.readline()))
                    with open(f"{basedir}/{dir_name}/X_{name}", "r") as f:
                        globals()[f"X_{d}d_q={q}_N={N}"] = (eval(f.readline()))
                    with open(f"{basedir}/{dir_name}/T_{name}", "r") as f:
                        globals()[f"T_{d}d_q={q}_N={N}"] = (eval(f.readline()))
                f = plt.figure(figsize=(18, 10))
                if h == 0:
                    sign = '='
                elif h > 0:
                    sign = '+'
                else:
                    sign = '-'
                suptitle = f"$q$={q}, $d$={d}, $h$={h}"
                f.suptitle(suptitle, fontsize=15, fontweight="bold")
                titles = ['Энергия', 'Намагниченность', 'Теплоёмкость', 'Магн. восприимчивость']
                for j in range(4):
                    title = titles[j]
                    letter = 'EMCX'[j]
                    labels = ['$E$', '$M$', '$C_v$', '$\chi$']
                    ax = plt.subplot(2, 2, j + 1)
                    for i in range(len(globals()[f"NN_{d}"])):
                        N = globals()[f"NN_{d}"][i]
                        plt.scatter(globals()[f"T_{d}d_q={q}_N={N}"], globals()[f"{letter}_{d}d_q={q}_N={N}"], s=5,
                                    label=f"N={N}", marker='x')
                    ax.set_xlabel(r"$\frac{k_BT}{|\;J\;|}$", fontsize=15, fontweight="bold")
                    if d == 2:
                        crit = round(abs(J) / (math.log(1 + q ** (1 / 2))), 3)
                        plt.axvline(x=crit, c='r', alpha=0.5, label=f"$T_c=${crit}")
                    label = str(labels[j])
                    ax.set_ylabel(f"{label}", fontsize=15, fontweight="bold")
                    ax.axis('tight')
                    ax.set_title(title, fontsize=20, fontweight="bold")
                    ax.grid('--', alpha=0.5)
                    f.tight_layout(pad=3.0)
                    ax.legend(loc='best')
                fig_name = f"plot_potts_{d}d_q={q}_h{sign}0.png"
                plt.savefig(f"{fig_name}", bbox_inches='tight', dpi=400)
                plt.show()
