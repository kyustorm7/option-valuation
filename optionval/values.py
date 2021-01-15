from math import log, sqrt, pi, exp
from scipy.stats import norm
import numpy as np
import pandas as pd

# C: 콜옵션 가격 / Call option price
# P: 풋옵션 가격/ Put option price
# N: 정규분포함수 / CDF of the normal distribution
# S: 주식 현물 가격 / Current value of underlying asset
# E: 옵션 행사가 / Exercise price
# r: 무위험 이자율 / Annual risk-free interest rate over the period from now to expiration date
# T: 총 년도수 / Time to expiration date(in years)
# sigma: 변동성 지수 / Standard deviation (per year) of continuous stock returns

#d1 d2값 계산
def d1(S, E, T, r, sigma):
    # Check Input for Value Error
    if not isinstance(S, (int, float)):
        raise ValueError(("S must be numeric"))
    if not isinstance(E, (int, float)):
        raise ValueError(("E must be numeric"))
    if not isinstance(T, (int, float)):
        raise ValueError(("T must be numeric"))
    if not isinstance(r, (int, float)):
        raise ValueError(("r must be numeric"))
    if not isinstance(sigma, (int, float)):
        raise ValueError(("sigma must be numeric"))
    if r < 0 or r > 1:
        raise ValueError(("r must be between 0 and 1"))

    # return d1 value used in blackscholes model
    return(log(S/E)+(r+sigma**2/2)*T)/sigma*sqrt(T)

def d2(S, E, T, r, sigma):
    # Check Input for Value Error
    if not isinstance(S, (int, float)):
        raise ValueError(("S must be numeric"))
    if not isinstance(E, (int, float)):
        raise ValueError(("E must be numeric"))
    if not isinstance(T, (int, float)):
        raise ValueError(("T must be numeric"))
    if not isinstance(r, (int, float)):
        raise ValueError(("r must be numeric"))
    if not isinstance(sigma, (int, float)):
        raise ValueError(("sigma must be numeric"))
    if r < 0 or r > 1:
        raise ValueError(("r must be between 0 and 1"))

    # return d2 value used in blackscholes model
    return d1(S,E,T,r,sigma)-sigma*sqrt(T)

def call_delta(S, E, T, r, sigma):
    # calculate delta in call option
    return norm.cdf(d1(S,E,T,r,sigma))
def call_gamma(S, E, T, r, sigma):
    return norm.pdf(d1(S,E,T,r,sigma))/(S*sigma*sqrt(T))
def call_vega(S, E, T, r, sigma):
    return 0.01*(S*norm.pdf(d1(S,E,T,r,sigma))*sqrt(T))
def call_theta(S, E, T, r, sigma):
    return 0.01*(-(S*norm.pdf(d1(S,E,T,r,sigma))*sigma)/(2*sqrt(T)) - r*E*exp(-r*T)*norm.cdf(d2(S,E,T,r,sigma)))
def call_rho(S, E, T, r, sigma):
    return 0.01*(E*T*exp(-r*T)*norm.cdf(d2(S,E,T,r,sigma)))


def blackscholes(S, E, T, r, sigma, PutCall = 'C'):
    if PutCall != 'C' and PutCall != 'P':
        raise ValueError(("Vairiable PutCall must be either 'C' or 'P'. "))
    if PutCall == 'C':  # Call Option
        result = S * norm.cdf(d1(S, E, T, r, sigma)) - E * exp(-r * T) * norm.cdf(d2(S, E, T, r, sigma))
    if PutCall == 'P':  # Put Option
        result = E * exp(-r * T) * norm.cdf(-d2(S, E, T, r, sigma)) - S * norm.cdf(-d1(S, E, T, r, sigma))

    return result

def volatility(stock_sd, bond_sd, stock_weight = None, bond_weight = None, corr):
    # Check for Value Error
    if not isinstance(stock_sd, (int, float)):
        raise ValueError(("stock_sd must be numeric"))
    if not isinstance(bond_sd, (int, float)):
        raise ValueError(("bond_sd must be numeric"))
    if not isinstance(stock_weight, (type(None), int, float)):
        raise ValueError(("stock_weight must be numeric"))
    if not isinstance(bond_weight, (type(None), int, float)):
        raise ValueError(("bond_weight must be numeric"))
    if not isinstance(corr, (int, float)):
        raise ValueError(("corr must be numeric"))
    if stock_weight == None and bond_weight = None:
        stock_weight, bond_weight = 0.5, 0.5
    elif stock_weight = None:
        stock_weight = 1 - bond_weight
    elif bond_weight = None:
        bond_weight = 1 - stock_weight
    if stock_weight < 0 or stock_weight > 1 or bond_weight < 0 or stock_weight > 1:
        raise ValueError(("Weight variables must be between 0 and 1."))
    if stock_weight + bond_weight != 1:
        raise ValueError(("Sum of stock wieght and bond_weight must be equal to 1."))
    if corr < 0 or corr > 1:
        raise ValueError(("Variable corr must be between 0 and 1."))


    result = stock_sd**2*stock_weight**2 + stock_sd**2*stock_weight**2 + 2*stock_weight*bond_weight*stock_sd*bond_sd*corr
    return sqrt(result)

