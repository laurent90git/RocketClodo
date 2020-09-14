## Simulation simplifié de la trajectoire d'une fusée
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
from model import modelfun

if __name__=='__main__':

    const_G=6.67408e-11
    Mterre = 5.972e24
    Rterre = 6.3e6
    grav_earth= lambda r : -Mterre*const_G/r**2
    SCx = np.pi*2**2 * 0.1
    
    pos0 = np.array([Rterre, 0.])
    v0 = np.array([0., 0.])
    m0 = np.array([750e3])

    y0 = np.concatenate((pos0,v0,m0))
    print('initial state: ',y0)
    
    
    def density(r):
      h = np.abs(r)-Rterre
      rho=1.3*np.exp(-h/7000.) # atmo isotherme
      return rho
    def fun_trainee(r,v):
        normeV = np.linalg.norm(v,axis=0)
        return -0.5*density(r)*SCx*v*(normeV)
    options= {'moteur' : {'debit' : lambda t : 200.*(t<200.0), 
                             've' : lambda t : 250000.},
              'gravite': {'champs': grav_earth},
              'aero'   : {'trainee': fun_trainee},
              'controle': {'orientation': lambda t: np.array([1,0])*(t<=100.) + np.array([0.5,0.5])*(t<200.)*(t>100.)}
              }
    
    
    out = scipy.integrate.solve_ivp(fun=lambda t,y: modelfun(t,y,options), t_span=[0.,250.], y0=y0,
       method='RK45', t_eval=None,   dense_output=False, events=None, vectorized=True, rtol=1e-4, atol=1e-4, max_step=np.inf)
    
    
    fig, ax = plt.subplots(y0.size,1,sharex=True)
    for i in range(y0.size):
      ax[i].plot(out['t'], out['y'][i,:], marker='.')
    plt.draw()
    
    
    print('done')
    
    
    r =  out['y'][0,:]
    theta =  out['y'][1,:]
    r_p =  out['y'][2,:]
    theta_p =  out['y'][3,:]
    m =  out['y'][4,:]
    
    v = np.vstack([r_p, theta_p*r])
    normeV= np.linalg.norm(v, axis=0)
    
    theta = out['y'][1,:]
    x, y = r*np.cos(theta), r*np.sin(theta)
    
    plt.figure()
    plt.plot(x,y)
    plt.plot(x[0],y[0], marker='o', color='r')
    plt.plot(x[-1],y[-1], marker='o', color='g')
    plt.plot(0,0, marker='*', color='b')
    
    
    fig, ax = plt.subplots(3,1,sharex=True)
    trainee = fun_trainee(r, v)
    ax[0].plot(out['t'], r)
    ax[0].set_ylabel('r')
    ax[1].plot(out['t'], normeV)
    ax[1].set_ylabel('||v||')
    ax[2].plot(out['t'], np.linalg.norm(trainee, axis=0))
    ax[2].set_ylabel('||trainee||')
    ax[-1].set_xlabel('t')
    for a in ax:
        a.grid()