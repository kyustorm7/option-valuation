from math import log, sqrt, pi, exp
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import numpy as np
import pandas as pd

# n = 총 단계수 / number of binomial steps
# S = 주식 초기 가격 / initial stock price
# K = 옵션 행사가 / strike price
# r = 무위험 이자율 / risk free interest rate per annum
# v = 변동성 지수 / volatility factor
# t = 총 년도수 / Time to expiration date(in years)
# Putcall = 옵션 종류 / Option Type

def BinomialAmerican(n, S, K, r, v, t, PutCall = "C"):

    # Check Input for ValueError
    if not isinstance(n, (int)):
        raise ValueError(("n must be integer"))
    if not isinstance(S, (int, float)):
        raise ValueError(("S must be numeric"))
    if not isinstance(K, (int, float)):
        raise ValueError(("K must be numeric"))
    if not isinstance(r, (int, float)):
        raise ValueError(("r must be numeric"))
    if not isinstance(v, (int, float)):
        raise ValueError(("v must be numeric"))
    if not isinstance(t, (int, float)):
        raise ValueError(("t must be numeric"))
    if r < 0 or r > 1:
        raise ValueError(("r must be between 0 and 1"))
    if PutCall != 'C' and PutCall != 'P':
        raise ValueError(("Vairiable PutCall must be either 'C' or 'P'. "))


    deltat = t / n
    u = np.exp(v * np.sqrt(deltat))
    d = 1. / u
    p = (np.exp(r * deltat) - d) / (u - d)

    # Binomial Tree에서 각 노드의 price 계산
    stockvalue = np.zeros((n + 1, n + 1))
    stockvalue[0, 0] = S
    for i in range(1, n + 1):
        stockvalue[i, 0] = stockvalue[i - 1, 0] * u
        for j in range(1, i + 1):
            stockvalue[i, j] = stockvalue[i - 1, j - 1] * d

    # Binomial Tree에서 각 노드의 payoff 계산
    optionvalue = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        if PutCall == "C":  # Call
            optionvalue[n, j] = max(0, stockvalue[n, j] - K)
        elif PutCall == "P":  # Put
            optionvalue[n, j] = max(0, K - stockvalue[n, j])

    # 옵션 가격 구하기
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            if PutCall == "P":
                optionvalue[i, j] = max(0, K - stockvalue[i, j], np.exp(-r * deltat) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1]))
            elif PutCall == "C":
                optionvalue[i, j] = max(0, stockvalue[i, j] - K, np.exp(-r * deltat) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1]))
    return optionvalue[0, 0]


def BinomialEuropean(n, S, K, r, v, t, PutCall):

    # Check Input for ValueError
    if not isinstance(n, (int)):
        raise ValueError(("n must be integer"))
    if not isinstance(S, (int, float)):
        raise ValueError(("S must be numeric"))
    if not isinstance(K, (int, float)):
        raise ValueError(("K must be numeric"))
    if not isinstance(r, (int, float)):
        raise ValueError(("r must be numeric"))
    if not isinstance(v, (int, float)):
        raise ValueError(("v must be numeric"))
    if not isinstance(t, (int, float)):
        raise ValueError(("t must be numeric"))
    if r < 0 or r > 1:
        raise ValueError(("r must be between 0 and 1"))
    if PutCall != 'C' and PutCall != 'P':
        raise ValueError(("Vairiable PutCall must be either 'C' or 'P'. "))

    deltat = t / n
    u = np.exp(v * np.sqrt(deltat))
    d = 1. / u
    p = (np.exp(r * deltat) - d) / (u - d)

    # Binomial Tree에서 각 노드의 price 계산
    stockvalue = np.zeros((n + 1, n + 1))
    stockvalue[0, 0] = S
    for i in range(1, n + 1):
        stockvalue[i, 0] = stockvalue[i - 1, 0] * u
        for j in range(1, i + 1):
            stockvalue[i, j] = stockvalue[i - 1, j - 1] * d

    # Binomial Tree에서 각 노드의 payoff 계산
    optionvalue = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        if PutCall == "C":  # Call
            optionvalue[n, j] = max(0, stockvalue[n, j] - K)
        elif PutCall == "P":  # Put
            optionvalue[n, j] = max(0, K - stockvalue[n, j])

    # 옵션 가격 구하기
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            if PutCall == "P":
                optionvalue[i, j] = np.exp(-r * deltat) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1])
            elif PutCall == "C":
                optionvalue[i, j] = np.exp(-r * deltat) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1])
    return optionvalue[0, 0]


def BinomialEuropean_graph(n, S, K, r, v, t, PutCall):

    # Check Input for ValueError
    if not isinstance(n, (int)):
        raise ValueError(("n must be integer"))
    if not isinstance(S, (int, float)):
        raise ValueError(("S must be numeric"))
    if not isinstance(K, (int, float)):
        raise ValueError(("K must be numeric"))
    if not isinstance(r, (int, float)):
        raise ValueError(("r must be numeric"))
    if not isinstance(v, (int, float)):
        raise ValueError(("v must be numeric"))
    if not isinstance(t, (int, float)):
        raise ValueError(("t must be numeric"))
    if r < 0 or r > 1:
        raise ValueError(("r must be between 0 and 1"))
    if PutCall != 'C' and PutCall != 'P':
        raise ValueError(("Vairiable PutCall must be either 'C' or 'P'. "))

    if PutCall == "C":
        y = [-BinomialEuropean(n, S, K, r, v, t, "C")] * (K)
        y += [x - BinomialEuropean(n, S, K, r, v, t, "C") for x in range(K)]

        plt.plot(range(2 * K), y, color='red')
        plt.axis([0, 2 * K, min(y) - 10, max(y) + 10])
        plt.xlabel('Value of Underlying asset')
        plt.ylabel('Payoff')
        plt.axvline(x=K, linestyle='--', color='black')
        plt.axhline(y=0, linestyle=':', color='black')
        plt.title('European Call Option')
        plt.show()

    elif PutCall == "P":
        z = [-x + K - BinomialEuropean(n, S, K, r, v, t, "P") for x in range(K)]
        z += [-BinomialEuropean(n, S, K, r, v, t, "P")] * (K)

        plt.plot(range(2 * K), z)
        plt.axis([0, 2 * K, min(z) - 10, max(z) + 10])
        plt.xlabel('Value of Underlying asset')
        plt.ylabel('Payoff')
        plt.axvline(x=K, linestyle='--', color='black')
        plt.axhline(y=0, linestyle=':', color='black')
        plt.title('European Put Option')
        plt.show()


def BinomialAmerican_graph(n, S, K, r, v, t, PutCall):
    # Check Input for ValueError
    if not isinstance(n, (int)):
        raise ValueError(("n must be integer"))
    if not isinstance(S, (int, float)):
        raise ValueError(("S must be numeric"))
    if not isinstance(K, (int, float)):
        raise ValueError(("K must be numeric"))
    if not isinstance(r, (int, float)):
        raise ValueError(("r must be numeric"))
    if not isinstance(v, (int, float)):
        raise ValueError(("v must be numeric"))
    if not isinstance(t, (int, float)):
        raise ValueError(("t must be numeric"))
    if r < 0 or r > 1:
        raise ValueError(("r must be between 0 and 1"))
    if PutCall != 'C' and PutCall != 'P':
        raise ValueError(("Vairiable PutCall must be either 'C' or 'P'. "))


    if PutCall == "C":
        y = [-BinomialAmerican(n, S, K, r, v, t, "C")] * (K)
        y += [x - BinomialAmerican(n, S, K, r, v, t, "C") for x in range(K)]

        plt.plot(range(2 * K), y, color='red')
        plt.axis([0, 2 * K, min(y) - 10, max(y) + 10])
        plt.xlabel('Value of Underlying asset')
        plt.ylabel('Payoff')
        plt.axvline(x=K, linestyle='--', color='black')
        plt.axhline(y=0, linestyle=':', color='black')
        plt.title('American Call Option')
        plt.show()

    elif PutCall == "P":
        z = [-x + K - BinomialAmerican(n, S, K, r, v, t, "P") for x in range(K)]
        z += [-BinomialAmerican(n, S, K, r, v, t, "P")] * (K)

        plt.plot(range(2 * K), z)
        plt.axis([0, 2 * K, min(z) - 10, max(z) + 10])
        plt.xlabel('Value of Underlying asset')
        plt.ylabel('Payoff')
        plt.axvline(x=K, linestyle='--', color='black')
        plt.axhline(y=0, linestyle=':', color='black')
        plt.title('American Put Option')
        plt.show()


def BinomialAmerican_tree(n, S, K, r, v, t, PutCall):
    # Check Input for ValueError
    if not isinstance(n, (int)):
        raise ValueError(("n must be integer"))
    if not isinstance(S, (int, float)):
        raise ValueError(("S must be numeric"))
    if not isinstance(K, (int, float)):
        raise ValueError(("K must be numeric"))
    if not isinstance(r, (int, float)):
        raise ValueError(("r must be numeric"))
    if not isinstance(v, (int, float)):
        raise ValueError(("v must be numeric"))
    if not isinstance(t, (int, float)):
        raise ValueError(("t must be numeric"))
    if r < 0 or r > 1:
        raise ValueError(("r must be between 0 and 1"))
    if PutCall != 'C' and PutCall != 'P':
        raise ValueError(("Vairiable PutCall must be either 'C' or 'P'. "))

    deltat = t / n
    u = np.exp(v * np.sqrt(deltat))
    d = 1. / u
    p = (np.exp(r * deltat) - d) / (u - d)

    # Binomial Tree에서 각 노드의 price 계산
    stockvalue = np.zeros((n + 1, n + 1))
    stockvalue[0, 0] = S
    for i in range(1, n + 1):
        stockvalue[i, 0] = stockvalue[i - 1, 0] * u
        for j in range(1, i + 1):
            stockvalue[i, j] = stockvalue[i - 1, j - 1] * d

    # Binomial Tree에서 각 노드의 payoff 계산
    optionvalue = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        if PutCall == "C":  # Call
            optionvalue[n, j] = max(0, stockvalue[n, j] - K)
        elif PutCall == "P":  # Put
            optionvalue[n, j] = max(0, K - stockvalue[n, j])

    # 옵션 가격 구하기
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            if PutCall == "P":
                optionvalue[i, j] = max(0, K - stockvalue[i, j], np.exp(-r * deltat) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1]))
            elif PutCall == "C":
                optionvalue[i, j] = max(0, stockvalue[i, j] - K, np.exp(-r * deltat) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1]))

    if PutCall == "C":
        fig = plt.figure(figsize=[7, 7])
        for i in range(n):
            x = [1, 0, 1]
            for j in range(i):
                x.append(0)
                x.append(1)
            x = np.array(x) + i
            y = np.arange(-(i + 1), i + 2)[::-1]
            for j in range(0, n + 1):
                for i in range(j, n + 1):
                    plt.figtext(0.15 + 0.7 / n * i, 0.54 + 0.34 / n * (i - 2 * j), round(optionvalue[i, j], 2))
                    plt.plot(x, y, 'b->', color='black')
                    plt.title('Call option price(American)')
                    plt.axis('off')
        plt.text(n + n / 3, 0 + n / 5 * 4, 'Strike Price = ' + str(K), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 3, 'Initial Stock Price = ' + str(S), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 2, 'p = ' + str(round(p, 3)), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 1, '1 - p = ' + str(1 - round(p, 3)), size=15)
        plt.show()

    elif PutCall == "P":
        fig = plt.figure(figsize=[7, 7])
        for i in range(n):
            x = [1, 0, 1]
            for j in range(i):
                x.append(0)
                x.append(1)
            x = np.array(x) + i
            y = np.arange(-(i + 1), i + 2)[::-1]
            for j in range(0, n + 1):
                for i in range(j, n + 1):
                    plt.figtext(0.15 + 0.7 / n * i, 0.54 + 0.34 / n * (i - 2 * j), round(optionvalue[i, j], 2))
                    plt.plot(x, y, 'b->', color='black')
                    plt.title('Put option price(American)')
                    plt.axis('off')
        plt.text(n + n / 3, 0 + n / 5 * 4, 'Strike Price = ' + str(K), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 3, 'Initial Stock Price = ' + str(S), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 2, 'p = ' + str(round(p, 3)), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 1, '1 - p = ' + str(1 - round(p, 3)), size=15)
        plt.show()


def BinomialEuropean_tree(n, S, K, r, v, t, PutCall):
    # Check Input for ValueError
    if not isinstance(n, (int)):
        raise ValueError(("n must be integer"))
    if not isinstance(S, (int, float)):
        raise ValueError(("S must be numeric"))
    if not isinstance(K, (int, float)):
        raise ValueError(("K must be numeric"))
    if not isinstance(r, (int, float)):
        raise ValueError(("r must be numeric"))
    if not isinstance(v, (int, float)):
        raise ValueError(("v must be numeric"))
    if not isinstance(t, (int, float)):
        raise ValueError(("t must be numeric"))
    if r < 0 or r > 1:
        raise ValueError(("r must be between 0 and 1"))
    if PutCall != 'C' and PutCall != 'P':
        raise ValueError(("Vairiable PutCall must be either 'C' or 'P'."))


    deltat = t / n
    u = np.exp(v * np.sqrt(deltat))
    d = 1. / u
    p = (np.exp(r * deltat) - d) / (u - d)

    # Binomial Tree에서 각 노드의 price 계산
    stockvalue = np.zeros((n + 1, n + 1))
    stockvalue[0, 0] = S
    for i in range(1, n + 1):
        stockvalue[i, 0] = stockvalue[i - 1, 0] * u
        for j in range(1, i + 1):
            stockvalue[i, j] = stockvalue[i - 1, j - 1] * d

    # Binomial Tree에서 각 노드의 payoff 계산
    optionvalue = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        if PutCall == "C":  # Call
            optionvalue[n, j] = max(0, stockvalue[n, j] - K)
        elif PutCall == "P":  # Put
            optionvalue[n, j] = max(0, K - stockvalue[n, j])

    # 옵션 가격 구하기
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            if PutCall == "P":
                optionvalue[i, j] = np.exp(-r * deltat) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1])
            elif PutCall == "C":
                optionvalue[i, j] = np.exp(-r * deltat) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1])

    if PutCall == "C":
        fig = plt.figure(figsize=[7, 7])
        for i in range(n):
            x = [1, 0, 1]
            for j in range(i):
                x.append(0)
                x.append(1)
            x = np.array(x) + i
            y = np.arange(-(i + 1), i + 2)[::-1]
            for j in range(0, n + 1):
                for i in range(j, n + 1):
                    plt.figtext(0.15 + 0.7 / n * i, 0.54 + 0.34 / n * (i - 2 * j), round(optionvalue[i, j], 2))
                    plt.plot(x, y, 'b->', color='black')
                    plt.title('Call option price(European)')
                    plt.axis('off')
        plt.text(n + n / 3, 0 + n / 5 * 4, 'Strike Price = ' + str(K), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 3, 'Initial Stock Price = ' + str(S), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 2, 'p = ' + str(round(p, 3)), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 1, '1 - p = ' + str(1 - round(p, 3)), size=15)
        plt.show()

    elif PutCall == "P":
        fig = plt.figure(figsize=[7, 7])
        for i in range(n):
            x = [1, 0, 1]
            for j in range(i):
                x.append(0)
                x.append(1)
            x = np.array(x) + i
            y = np.arange(-(i + 1), i + 2)[::-1]
            for j in range(0, n + 1):
                for i in range(j, n + 1):
                    ax = plt.figtext(0.15 + 0.7 / n * i, 0.54 + 0.34 / n * (i - 2 * j), round(optionvalue[i, j], 2))
                    plt.plot(x, y, 'b->', color='black')
                    plt.title('Put option price(European)')
                    plt.axis('off')
        plt.text(n + n / 3, 0 + n / 5 * 4, 'Strike Price = ' + str(K), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 3, 'Initial Stock Price = ' + str(S), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 2, 'p = ' + str(round(p, 3)), size=15)
        plt.text(n + n / 3, 0 + n / 5 * 1, '1 - p = ' + str(1 - round(p, 3)), size=15)
        plt.show()