## Simulation simplifié de la trajectoire d'une fusée
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
from model import modelfun, getVarsFromState

if __name__=='__main__':
    m_shuttle= np.array([[0,  2051113.],
                    [10,  1935155],
                    [20,  1799290],
                    [30,  1681120],
                    [40,  1567611],
                    [50,  1475282],
                    [60,  1376301],
                    [70,  1277921],
                    [80,  1177704],
                    [90,  1075683],
                    [100,  991872],
                    [110,  913254],
                    [120,  880377],
                    ])
    plt.figure()
    plt.plot(m_shuttle[:,0],m_shuttle[:,1])
    
    debit = np.diff(m_shuttle[:,1])/ np.diff(m_shuttle[:,0])
    ve=2500.
    accel_en_g = (-debit*2500.)/(9.81*m_shuttle[:-1,1])

    
    const_G=6.67408e-11
    Mterre = 5.972e24
    Rterre = 6.3e6
    grav_earth= lambda r : -Mterre*const_G/r**2
    SCx = 10*np.pi*2*30 * 0.1
    
    #pos0 = np.array([Rterre+1., 0.])    
    pos0 = np.array([Rterre, 0.])

    v0 = np.array([0., 0.])
    m0 = np.array([1900e3])
    

    y0 = np.concatenate((pos0,v0,m0))
    print('initial state: ',y0)
    
    def event_mass_null(t,y):
        return y[4]
      
    def event_altitude_null(t,y):
        return y[0]-Rterre+1.

    event_altitude_null.terminal = True
    event_mass_null.terminal = True
    
    def density(r):
      h = np.abs(r)-Rterre
      rho=1.3*np.exp(-h/7000.) # atmo isotherme
      return rho
    def fun_trainee(r,v):
        normeV = np.linalg.norm(v,axis=0)
        return -0.5*density(r)*SCx*v*(normeV)
    options= {'moteur' : {'debit' : lambda t : 12000.*(t<200.0), 
                             've' : lambda t : 2500.},
              'gravite': {'champs': grav_earth},
              'aero'   : {'trainee': fun_trainee},
              'controle': {'orientation': lambda t: np.array([1,0])*(t<=100.) + np.array([0.5,0.5])*(t>100.)},
              'structure': {'masse_a_vide':70e3}
              }
    
    def processRestart(out):
      """Fonction qui analyse les évenements et prépare la suite du calcul en conséquence"""
      # trouver quel évenement a été trigger
      try:
          i_event = np.where([np.size(t)>0 for t in out.t_events])[0][0]
      except IndexError:
          i_event=None
         
      t0 = out.t[-1]
      y0 = out.y[:,-1]
      r, theta, r_p, theta_p, m = getVarsFromState(y0, options)
      bContinue=True
      
      if i_event is None:
        message = "fin du temps d'intégration demandé"
        bContinue = False
      if i_event==0: # masse nulle
        options["moteur"]["debit"] = lambda t: 0. # plus de poussée
        message = "masse nulle"
      elif i_event==1: # altitude nulle
        message = 'altitude nulle !'
        bContinue = False
      return bContinue, message, t0, y0, options

    t0 = 0.
    out = []
    bContinue=True

    while bContinue:
      out.append( scipy.integrate.solve_ivp(fun=lambda t,y: modelfun(t,y,options), t_span=[t0,t0+5000.],    y0=y0,
        method='RK45', t_eval=None,   dense_output=False, events=[event_mass_null,event_altitude_null], vectorized=True, rtol=1e-7, atol=1e-4, max_step=np.inf) )
      bContinue, message, t0, y0, options = processRestart(out[-1])
      print(message)
    
    fig, ax = plt.subplots(y0.size,1,sharex=True)
    for i in range(y0.size):
      for j in range(len(out)):
        ax[i].plot(out[j]['t'], out[j]['y'][i,:], marker='.')
    plt.draw()
    

    print('done')
    
    
    temp = np.linspace(0,2*np.pi,1000)
    terrex, terrey = Rterre*np.cos(temp), Rterre*np.sin(temp)
    
    plt.figure()
    plt.plot(terrex, terrey, marker=None, color=[0,0,0], linestyle='--')
    
    for j in range(len(out)):
        r, theta, r_p, theta_p, m = getVarsFromState(out[j].y, options)
        g = options['gravite']['champs'](r)
        t = out[j]['t']
    
        v = np.vstack([r_p, theta_p*r])
        normeV= np.linalg.norm(v, axis=0)
        
        x, y = r*np.cos(theta), r*np.sin(theta)
        plt.plot(x,y)
        plt.plot(x[0],y[0], marker='o', color='r')
        plt.plot(x[-1],y[-1], marker='*', color='g')
    
    plt.plot(0,0, marker='*', color='b')
    
    
    fig, ax = plt.subplots(4,1,sharex=True)
    trainee = fun_trainee(r, v)
    ax[0].plot(t, (r-Rterre)/1e3)
    ax[0].set_ylabel('h (km)')
    ax[1].plot(t, normeV)
    ax[1].set_ylabel('||v||')
    ax[2].plot(t, np.linalg.norm(trainee, axis=0)/(m*g))
    ax[2].set_ylabel('||trainee|| (en g)')
    ax[3].plot(t,m)
    ax[3].set_ylabel('m')
    ax[-1].set_xlabel('t')
    for a in ax:
        a.grid()