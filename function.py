## Simulation simplifié de la trajectoire d'une fusée
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

pos0 = np.array([1., 0.])
v0 = np.array([0., 1.25])
m0 = np.array([1.])

y0 = np.concatenate((pos0,v0,m0))

print('initial state: ',y0)

const_G=1.

grav_earth= lambda r : -1.0*const_G/r**2

options= {'moteur' : {'debit' : lambda t : 0*0.01*(t<1.), 
                         've' : lambda t : 400.},
          'gravite': {'champs': grav_earth},
          'aero'   : {'trainee':lambda x,v: -0*0.001*v},
          'controle': {'orientation': lambda t: np.array([0,1])}
          }

def modelfun(t,y,options):
  # récupérer les états
  r = y[0,:]
  theta = y[1,:]
  r_p = y[2,:]
  theta_p = y[3,:]
  m = y[4,:]

  # données de la fusée
  normeV = np.sqrt(r_p**2 + (r*theta_p)**2)
  debit = options['moteur']['debit'](t)
  ve = options['moteur']['ve'](t)
  g = options['gravite']['champs'](r)
  trainee = options['aero']['trainee'](r,normeV)
  orientation = options['controle']['orientation'](t)
  normeO = np.linalg.norm(orientation)
  orientation = orientation/normeO # renormaliser

  # définir les dérivées temporelles
  dydt = np.zeros_like(y)
  dydt[4,:] = -debit
  dydt[0,:] = r_p
  dydt[1,:] = r*theta_p
  dydt[2,:] = (m*g + (debit*ve+trainee)*orientation[0]/normeO + r*theta_p**2)/m
  dydt[3,:] = ((debit*ve + trainee)*orientation[1]/normeO - 2*r_p*theta_p) /(m*r)
  return dydt

out = scipy.integrate.solve_ivp(fun=lambda t,y: modelfun(t,y,options), t_span=[0.,100.], y0=y0,
   method='LSODA', t_eval=None,   dense_output=False, events=None, vectorized=True, rtol=1e-8, atol=1e-8, max_step=1e-2)


fig, ax = plt.subplots(y0.size,1,sharex=True)
for i in range(y0.size):
  ax[i].plot(out['t'], out['y'][i,:], marker='.')
plt.draw()


print('done')


plt.figure()
r = out['y'][0,:]
theta = out['y'][1,:]
x, y = r*np.cos(theta), r*np.sin(theta)

plt.plot(x,y)
plt.plot(x[0],y[0], marker='+', color='r')
plt.plot(x[-1],y[-1], marker='+', color='g')
plt.plot(0,0, marker='*', color='b')
