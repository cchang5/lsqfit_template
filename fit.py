import h5py as h5
import lsqfit
import gvar as gv
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    f = h5.File(filename)
    data = f['pi_p']['cl3_48_64_b6p1_m0p2450']['ml-0.2450_ms-0.2450']['corr'].value
    return data

def plot_data(data):
    x = np.arange(len(data))
    # plot effective mass
    meff = np.log(data/np.roll(data,-1))
    y = [i.mean for i in meff]
    err = [i.sdev for i in meff]
    fig = plt.figure(figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    ax.errorbar(x=x,y=y,yerr=err,ls='None',marker='o',fillstyle='none',markersize='5',elinewidth=1,capsize=2)
    ax.set_xlim([1,13])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('$t$', fontsize=20)
    ax.set_ylabel('$M^{eff}$', fontsize=20)
    plt.draw()
    plt.show()
    # plot scaled correlator
    scorr = data*np.exp(meff*np.arange(len(data)))
    y = [i.mean for i in scorr]
    err = [i.sdev for i in scorr]
    ax = plt.axes([0.15,0.15,0.8,0.8])
    ax.errorbar(x=x,y=y,yerr=err,ls='None',marker='o',fillstyle='none',markersize='5',elinewidth=1,capsize=2)
    ax.set_xlim([1,13])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('$t$', fontsize=20)
    ax.set_ylabel('$A^{eff}$', fontsize=20)
    plt.draw()
    plt.show()

class fit_functions():
    def __init__(self,nstates,T):
        self.nstates = nstates
        self.T = T
        return None
    def A(self,n,p):
        return p['A%s' %n]
    def E(self,n,p):
        E = p['E0']
        for ns in range(1,n+1):
            E += np.exp(p['E%s' %ns])
        return E
    def twopt(self,t,p):
        r = 0
        for n in range(self.nstates):
            An = self.A(n,p)
            En = self.E(n,p)
            r += An * (np.exp(-En*t) + np.exp(-En*(self.T-t)))
        return r

def priors(nstates):
    p = dict()
    p['E0'] = [0.6, 0.2]
    p['A0'] = [9.2E-3, 4.5E-3]
    p['E1'] = [-1.0, 1.0]
    p['A1'] = [0.0, 9.2E-3]
    prior = dict()
    for n in range(nstates):
        for k in p.keys():
            if int(k[-1]) == n:
                prior[k] = gv.gvar(p[k][0], p[k][1])
            else: pass
    return prior

def fit_data(data):
    nstates = 2
    x = np.arange(5,10) 
    y = data[x]
    p = priors(nstates)
    fitc = fit_functions(T=len(data)+1,nstates=nstates)
    fit = lsqfit.nonlinear_fit(data=(x,y),prior=p,fcn=fitc.twopt)
    print fit
    return fit

if __name__=='__main__':
    data = read_data('./p0_rxi.small.h5')
    corr0 = gv.dataset.avg_data(np.squeeze(data[:,:,0]))
    corr1 = gv.dataset.avg_data(np.squeeze(data[:,:,1]))
    #plot_data(corr0)
    fit = fit_data(corr0)
