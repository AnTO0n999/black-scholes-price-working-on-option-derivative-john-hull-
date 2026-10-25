import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import cmath

class BLCK_SCHLS:
    def __init__(self, s, k, t, sig, r):
        self.s = s
        self.k = k
        self.t = t
        self.sig = sig
        self.r = r

    def d1(self):
        ln_S_div_K = np.log(self.s) - np.log(self.k)
        d1 = ((ln_S_div_K + ((self.r + (self.sig**2)/2) * self.t)) / (self.sig * np.sqrt(self.t)))
        return d1

    def d2(self):
        d2 = self.d1() - (self.sig * np.sqrt(self.t))
        return d2

    def call_price(self):
        C = (self.s * norm.cdf(self.d1())) - (self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2()))
        return C

    def put_price(self):
        P = (self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2())) - (self.s * norm.cdf(-self.d1()))
        return P

#greeks
#delta
    def delta_put(self):
        del_p = norm.cdf(self.d1()) - 1
        return del_p

    def delta_call(self):
        del_c = norm.cdf(self.d1())
        return del_c

#theta
    def theta_put(self):
        Npri =  (1 / np.sqrt(2* np.pi)) * (np.exp(((-self.d1() ** 2 )/ 2)))
        theta_p = ((self.s * Npri * self.sig) / (2 * np.sqrt(self.t))) + (self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2()))
        per_trad_day_p = theta_p / 252
        return theta_p

    def theta_call(self):
        Npri =  (1 / np.sqrt(2 * np.pi )) * (np.exp(((-self.d1() ** 2 )/ 2)))
        theta_c = (-(self.s * Npri * self.sig) / (2 * np.sqrt(self.t))) - (self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2()))
        per_trad_day_c = theta_c / 252
        return theta_c

    def gamma(self):
        Npri = (1 / np.sqrt(2 * np.pi)) * (np.exp(((-self.d1() ** 2) / 2)))
        gam = Npri / (self.s * self.sig * np.sqrt(self.t))
        return gam

    def vega(self):
        Npri = (1 / np.sqrt(2 * np.pi)) * (np.exp(((-self.d1() ** 2) / 2)))
        vega = self.s * np.sqrt(self.t) * Npri
        return vega

    def rho_put(self):
        rho_put = -self.k * self.t * np.exp(-self.r * self.t) * norm.cdf(-self.d2())
        return rho_put

    def rho_call(self):
        rho_call = self.k * self.t * np.exp(-self.r * self.t) * norm.cdf(self.d2())
        return rho_call

option = BLCK_SCHLS(49, 50, 0.3846, 0.20, 0.05)
print(f"{option.d1():.4f}")
print(f"{option.d2():.4f}")
print(f"{option.call_price():.4f}")
print(f"{option.put_price():.4f}")
print(f"{option.delta_put():.4f}")
print(f"{option.delta_call():.4f}")
print(f"{option.theta_put():.4f}")
print(f"{option.theta_call():.4f}")
print(f"{option.gamma():.4f}")
print(f"{option.vega():.4f}")
print(f"{option.rho_put():.4f}")
print(f"{option.rho_call():.4f}")
