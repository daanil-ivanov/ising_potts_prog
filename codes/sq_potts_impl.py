import shutil
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import time
import multiprocessing
import os
from datetime import datetime
import zipfile
plt.rcParams.update({"font.family": "serif", "font.cursive": ["Comic Neue", "Comic Sans MS"], })

n_proc = multiprocessing.cpu_count()
eqSteps = int(5e2)
mcSteps = int(2e3)
it = 600
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)
NN_2 = [10, 20, 25, 30, 35]
NN_3 = [10, 20, 25, 30]
D = [2, 3]
Q = [4, 5]
T_2 = np.linspace(0.2, 1.6, nt)
T_3 = np.linspace(1., 3., nt)
J = 1
h = 0
p = 0.8


def mcmove_2d(config, beta, n, J, q, h, config_i):
    _ = 0
    while _ < n ** 2:
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        if config_i[i, j] == 1:
            _ += 1
            cur = config[i, j]
            new = np.random.randint(0, q)
            if new == cur:
                while new == cur:
                    new = np.random.randint(0, q)
            sumstates = np.zeros(q)
            newsum = np.zeros(q)
            for __ in range(0, 3, 2):
                if config_i[(i + (__ - 1)) % n, j]:
                    if config[(i + (__ - 1)) % n, j] == cur:
                        sumstates[cur] += 1
                    if config[(i + (__ - 1)) % n, j] == new:
                        newsum[new] += 1
            for __ in range(0, 3, 2):
                if config_i[i, (j + (__ - 1)) % n]:
                    if config[i, (j + (__ - 1)) % n] == cur:
                        sumstates[cur] += 1
                    if config[i, (j + (__ - 1)) % n] == new:
                        newsum[new] += 1
            dE = ((sum(newsum) - sum(sumstates)) * J * -1) - h * calcMagn_2d(config, n, q, config_i)
            if dE < 0 or rand() < np.exp(-dE * beta):
                config[i, j] = new
    return config


def mcmove_3d(config, beta, n, J, q, h, config_i):
    _ = 0
    while _ < n ** 3:
        i, j, k = np.random.randint(0, n), np.random.randint(0, n), np.random.randint(0, n)
        if config_i[i, j, k] == 1:
            _ += 1
            cur = config[i, j, k]
            new = np.random.randint(0, q)
            if new == cur:
                while new == cur:
                    new = np.random.randint(0, q)
            sumstates = np.zeros(q)
            newsum = np.zeros(q)
            for __ in range(0, 3, 2):
                if config_i[(i + (__ - 1)) % n, j, k]:
                    if config[(i + (__ - 1)) % n, j, k] == cur:
                        sumstates[cur] += 1
                    if config[(i + (__ - 1)) % n, j, k] == new:
                        newsum[new] += 1
            for __ in range(0, 3, 2):
                if config_i[i, (j + (__ - 1)) % n, k]:
                    if config[i, (j + (__ - 1)) % n, k] == cur:
                        sumstates[cur] += 1
                    if config[i, (j + (__ - 1)) % n, k] == new:
                        newsum[new] += 1
            for __ in range(0, 3, 2):
                if config_i[i, j, (k + (__ - 1)) % n]:
                    if config[i, j, (k + (__ - 1)) % n] == cur:
                        sumstates[cur] += 1
                    if config[i, j, (k + (__ - 1)) % n] == new:
                        newsum[new] += 1
            dE = (sum(newsum) - sum(sumstates)) * J * -1
            if h != 0:
                dE += - h * calcMagn_3d(config, n, q, config_i)
            if dE < 0 or rand() < np.exp(-dE * beta):
                config[i, j, k] = new
    return config


def calcEnergy_2d(config, n, J, q, h, config_i):
    sumstates = np.zeros(q)
    for i in range(n):
        for j in range(n):
            for _ in range(0, 3, 2):
                if config_i[(i + (_ - 1)) % n, j]:
                    if config[(i + (_ - 1)) % n, j] == config[i, j]:
                        sumstates[config[i, j]] += 1
            for _ in range(0, 3, 2):
                if config_i[i, (j + (_ - 1)) % n]:
                    if config[i, (j + (_ - 1)) % n] == config[i, j]:
                        sumstates[config[i, j]] += 1
    energy = sum(sumstates) / 2 * J * (-1)
    if h != 0:
        energy += calcMagn_2d(config, n, q, config_i) * h
    return energy


def calcEnergy_3d(config, n, J, q, h, config_i):
    sumstates = np.zeros(q)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for _ in range(0, 3, 2):
                    if config_i[(i + (_ - 1)) % n, j, k]:
                        if config[(i + (_ - 1)) % n, j, k] == config[i, j, k]:
                            sumstates[config[i, j, k]] += 1
                for _ in range(0, 3, 2):
                    if config_i[i, (j + (_ - 1)) % n, k]:
                        if config[i, (j + (_ - 1)) % n, k] == config[i, j, k]:
                            sumstates[config[i, j, k]] += 1
                for _ in range(0, 3, 2):
                    if config_i[i, j, (k + (_ - 1)) % n]:
                        if config[i, j, (k + (_ - 1)) % n] == config[i, j, k]:
                            sumstates[config[i, j, k]] += 1
    energy = sum(sumstates) / 2 * J * (-1)
    if h != 0:
        energy += (calcMagn_3d(config, n, q, config_i) * h)
    return energy


def calcMagn_2d(config, n, q, config_i):
    states = np.zeros(q)
    N = np.sum(config_i)
    for i in range(n):
        for j in range(n):
            if config_i[i, j]:
                states[config[i, j]] += 1 / N
    sq_sum = 0
    for _ in range(q):
        sq_sum += states[_] ** 2
    m = q / (q - 1) * (sq_sum - 1 / q)
    return m * N


def calcMagn_3d(config, n, q, config_i):
    states = np.zeros(q)
    N = np.sum(config_i)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if config_i[i, j, k]:
                    states[config[i, j, k]] += 1 / N
    sq_sum = 0
    for _ in range(q):
        sq_sum += states[_] ** 2
    m = q / (q - 1) * (sq_sum - 1 / q)
    return m * N


def potts_imp(calc, proc, N, d, J, q, h, p):
    flag = True
    if proc == 0:
        start = time.time()
        flag = False
    par = np.zeros(4 * calc).reshape((4, calc))
    for i in range(calc):
        par[0][i], par[1][i], par[2][i], par[3][i] = sim_tt(N, (calc * proc + i), d, J, q, h, p)
        if flag == False:
            flag = True
            left = ((time.time() - start) * ((100 / (int((1 / calc) * 10000) / 100)) - 1) / 60)
            print(
                f"\n{left:.2f} minutes left for №{globals()[f'NN_{d}'].index(N) + 1}/{len(globals()[f'NN_{d}'])} test in {d}d q={q} N={N} p={p}.")
            print(f"сurrent time = {datetime.now().strftime('%H:%M:%S')}")
    for i in range(4):
        file = open(f"{'EMCX'[i]}_{proc}.txt", "w")
        file.write(str(par[i].tolist()))
        file.close()


def sim_tt(N, tt, d, J, q, h, p):
    beta = abs(J) / globals()[f'T_{d}'][tt]
    Ene = np.array(np.zeros(mcSteps), dtype=np.longdouble)
    Mag = np.array(np.zeros(mcSteps), dtype=np.longdouble)
    if d == 2:
        config = np.random.randint(q, size=(N, N))
        config_i = np.ones((N, N))
        for i in range(N):
            for j in range(N):
                config_i[i, j] = np.random.choice(2, 1, p=[1 - p, p])
    else:
        config = np.random.randint(q, size=(N, N, N))
        config_i = np.ones((N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    config_i[i, j, k] = np.random.choice(2, 1, p=[1 - p, p])
    calcEnergy = f"calcEnergy_{d}d"
    mcmove = f"mcmove_{d}d"
    calcMagn = f"calcMagn_{d}d"
    for i in range(eqSteps):
        config = eval(mcmove)(config, beta, N, J, q, h, config_i)
    for i in range(mcSteps):
        config = eval(mcmove)(config, beta, N, J, q, h, config_i)
        Ene[i] = eval(calcEnergy)(config, N, J, q, h, config_i)
        Mag[i] = eval(calcMagn)(config, N, q, config_i)
    E_mean, M_mean, C, X = np.mean(Ene), np.mean(Mag), beta ** 2 * np.std(Ene) ** 2, beta * np.std(Mag) ** 2
    return E_mean / np.sum(config_i), M_mean / (np.sum(config_i)), C / np.sum(config_i), X / np.sum(config_i)


def processed(procs, calc, N, d, J, q, h, P):
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=potts_imp, args=(calc, proc, N, d, J, q, h, P))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    Start = time.time()
    for q in Q:
        for d in D:
            if (d == 2 and q == 5) or (d == 3 and q == 4):
                dir_name = f'data_sq_{d}d_q={q}_p={p}_potts'
                output_filename = f'data_sq_{d}d_q={q}_p={p}_potts'
                basedir = os.path.abspath(os.getcwd())
                if os.path.exists(dir_name):
                    shutil.rmtree(dir_name)
                os.mkdir(dir_name)
                for N in globals()[f'NN_{d}']:
                    processed(n_proc, calc, N, d, J, q, h, p)
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
                    for i in range(5):
                        file = open(f"{basedir}/{dir_name}/{'EMCXT'[i]}_{d}d_q={q}_p={p}_N={N}.txt", "w")
                        file.write(str([E, M.tolist(), C, X, globals()[f'T_{d}'].tolist()][i]))
                        file.close()
                shutil.make_archive(output_filename, 'zip', dir_name)
                shutil.rmtree(dir_name)
                print(f'\ntotal time {((time.time() - Start) / 60):.2f} minutes')
                dir_name = f'data_sq_{d}d_q={q}_p={p}_potts'
                basedir = os.path.abspath(os.getcwd())
                zip_name = f'data_sq_{d}d_q={q}_p={p}_potts.zip'
                with zipfile.ZipFile(f"{zip_name}", 'r') as zip_ref:
                    zip_ref.extractall(f"{basedir}/{dir_name}")
                # for N in globals()[f"NN_{d}"]:
                #     name = f"{d}d_q={q}_p={p}_N={N}.txt"
                #     with open(f"{basedir}/{dir_name}/E_{name}", "r") as f:
                #         globals()[f"E_{d}d_q={q}_p={p}_N={N}"] = (eval(f.readline()))
                #     with open(f"{basedir}/{dir_name}/M_{name}", "r") as f:
                #         globals()[f"M_{d}d_q={q}_p={p}_N={N}"] = abs(np.array(eval(f.readline()))).tolist()
                #     with open(f"{basedir}/{dir_name}/C_{name}", "r") as f:
                #         globals()[f"C_{d}d_q={q}_p={p}_N={N}"] = (eval(f.readline()))
                #     with open(f"{basedir}/{dir_name}/X_{name}", "r") as f:
                #         globals()[f"X_{d}d_q={q}_p={p}_N={N}"] = (eval(f.readline()))
                #     with open(f"{basedir}/{dir_name}/T_{name}", "r") as f:
                #         globals()[f"T_{d}d_q={q}_p={p}_N={N}"] = (eval(f.readline()))
                # f = plt.figure(figsize=(18, 10))
                # if h == 0:
                #     sign = '='
                # elif h > 0:
                #     sign = '+'
                # else:
                #     sign = '-'
                # suptitle = f"$q$={q}, $d$={d}, $h$={h}, $p$={p}"
                # f.suptitle(suptitle, fontsize=15, fontweight="bold")
                # titles = ['Энергия', 'Намагниченность', 'Теплоёмкость', 'Магн. восприимчивость']
                # for j in range(4):
                #     title = titles[j]
                #     letter = 'EMCX'[j]
                #     labels = ['$E$', '$M$', '$C_v$', '$\chi$']
                #     ax = plt.subplot(2, 2, j + 1)
                #     for i in range(len(globals()[f"NN_{d}"])):
                #         N = globals()[f"NN_{d}"][i]
                #         plt.scatter(globals()[f"T_{d}d_q={q}_p={p}_N={N}"],
                #                     globals()[f"{letter}_{d}d_q={q}_p={p}_N={N}"], s=8, label=f"N={N}")
                #     ax.set_xlabel(r"$\frac{k_BT}{|\;J\;|}$", fontsize=15, fontweight="bold")
                #     # crit = round(abs(J) / (math.log(1 + q ** (1 / 2))), 3)
                #     # plt.axvline(x=crit, c='r', alpha=0.5, label=f"$T_c=${crit}")
                #     label = str(labels[j])
                #     ax.set_ylabel(f"{label}", fontsize=15, fontweight="bold")
                #     ax.axis('tight')
                #     ax.set_title(title, fontsize=20, fontweight="bold")
                #     ax.grid('--', alpha=0.5)
                #     f.tight_layout(pad=3.0)
                #     ax.legend(loc='best')
                # fig_name = f"plot_potts_{d}d_q={q}_h{sign}0_p={p}.png"
                # plt.savefig(f"{fig_name}", bbox_inches='tight', dpi=400)
                # plt.show()
