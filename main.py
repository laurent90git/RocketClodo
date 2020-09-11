## Simulation simplifié de la trajectoire d'une fusée
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

pos0 = np.array([3., 0.])
v0 = np.array([0., 0.1])
m0 = np.array([1.])

y0 = np.concatenate((pos0,v0,m0))
# y0 = np.array([ 2.94188784, 39.59457074,  0.3992208 ,  0.16427853,  0.99      ])

print('initial state: ',y0)

const_G=1.

grav_earth= lambda r : -1.0*const_G/r**2

# TODO: atmosphère, fusée terre réaliste, optimisation trajectoire


def fun_trainee(r,v):
    normeV = np.linalg.norm(v,axis=0)
    return -0.01*v*(normeV)*np.exp(-0.1*r)

options= {'moteur' : {'debit' : lambda t : 0*0.001*(t<10.0), 
                         've' : lambda t : 100.},
          'gravite': {'champs': grav_earth},
          'aero'   : {'trainee': fun_trainee},
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
  trainee = options['aero']['trainee'](r,np.array([r_p, theta_p*r]))
  orientation = options['controle']['orientation'](t)
  normeO = np.linalg.norm(orientation)
  orientation = orientation/normeO # renormaliser

  # définir les dérivées temporelles
  dydt = np.zeros_like(y)
  dydt[4,:] = -debit
  dydt[0,:] = r_p
  dydt[1,:] = theta_p
  dydt[2,:] = (m*g + debit*ve*orientation[0] + trainee[0])/m + r*theta_p**2
  dydt[3,:] = ((debit*ve*orientation[1] + trainee[1])/m - 2*r_p*theta_p) /r
  # dydt[2,:] = (m*g + r*theta_p**2)/m
  # dydt[3,:] = (-2.*r_p*theta_p) /(m*r)
  return dydt

out = scipy.integrate.solve_ivp(fun=lambda t,y: modelfun(t,y,options), t_span=[0.,50.], y0=y0,
   method='BDF', t_eval=None,   dense_output=False, events=None, vectorized=True, rtol=1e-10, atol=1e-10, max_step=1e-1)


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