## Simulation simplifié de la trajectoire d'une fusée
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

# TODO: atmosphère, fusée terre réaliste, optimisation trajectoire

def getVarsFromState(y,options):
  if len(y.shape)==1:
    y = y[:,np.newaxis]
  r = y[0,:]
  theta = y[1,:]
  r_p = y[2,:]
  theta_p = y[3,:]
  m = y[4,:] + options['structure']['masse_a_vide']
  return r, theta, r_p, theta_p, m

def modelfun(t,y,options):
  # récupérer les états
  r, theta, r_p, theta_p, m = getVarsFromState(y=y,options=options)
  
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