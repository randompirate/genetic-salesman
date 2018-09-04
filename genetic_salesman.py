"""
  genetic_salesman.py
  Heuristic travelling salesman solver using genetic optimisation

  Tom Nijhuis 2018
"""


import random as rng
import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(1)

class NodeList(object):
  """
    A set of nodes implemented with complex numbers.
    Plotted in a rough circle
    The first element is always (0,0)
  """
  def __init__(self, count, noise_rate = .45):
    """Initialisation. Sets the xy coords of the nodes in a vector self.P
    """
    self.count = count

    angle = np.arange(count)/count * 2*np.pi * 1j
    noise = noise_rate*(1-2*np.random.random(count) + 1j*(1-2*np.random.random(count)))

    self.P = np.exp(angle) + noise



  def path_length(self, paths):
    """ The total distance for given paths (as rows) indicated as a list of indices.
      Wraps around to first point
    """
    # Wraparound:
    paths = np.hstack([paths[:,:], paths[:,0:1]])

    P_order = self.P[paths]
    dists = np.absolute(P_order[:,1:]-P_order[:,:-1])

    return np.sum(dists, axis=1, keepdims = True)


  def plot(self, ax, path = None):
    """Plotting the nodes and optionally a path on a 2d grid
    """
    path = np.hstack([path, path[0:1]])

    x, y = self.P.real, self.P.imag
    ax.plot(x,y, 'o')
    ax.plot(x[path], y[path], '-')


class GeneticOptimiser(object):
  """Genetic optimiser for TSP
    self.population contains gentypes (one per row)
  """
  def __init__(self, gentype_lngth, popsize = 1000, cost_fun = None):
    self.popsize = popsize
    self.gentype_lngth = gentype_lngth
    self.cost_fun = cost_fun

    self.costs = np.empty(self.popsize).reshape((self.popsize, 1))
    self.age = 0

    self.params = {
        'perc_top'        : .003,
        'max_crossover'   : .5,
        'tournament_size' : 10,

    }

    self.initialise_population()
    self.cost_me()



  def initialise_population(self):
    """Initialise the whole population
    every tour starts at 0
    """
    #Permute points 1 to k
    self.population = np.array([np.random.permutation(list(range(1, self.gentype_lngth)))
                                for i in range(self.popsize)])
    # Add zero to front
    self.population = np.hstack([np.zeros((self.popsize, 1), dtype = int), self.population])



  def evolve(self):
    """Perform one optimisation step on the whole population
    Mutation and crossover
    """
    top_index   = list(range(int(self.popsize * self.params['perc_top'])))
    # lucky_index = [s for s in np.random.permutation(range(int(self.popsize)))[:len(top_index)]]
    lucky_index = list(np.random.choice(range(self.popsize), size=len(top_index), replace = False))

    mutagen = np.random.choice(['randswap', 'pickinsert', 'neighswap', 'partreverse'])
    newpop = np.vstack([
        # Mutate
        self.mutate(top_index, mutagen),
        self.mutate(lucky_index, 'pickinsert'), #pickinsert is the better choice for poor agents
        # Crossover
        self.crossover(top_index, np.random.permutation(top_index)),
        self.crossover(lucky_index, top_index),

      ])


    # Add new members
    oldsize = self.popsize
    self.add_to_population(newpop)
    self.trim_population(oldsize)

    self.age += 1




  def add_to_population(self, newpop):
    """Introduce new agents to an existing population
    """

    # Add to population and cost array
    self.population = np.vstack([self.population, newpop])
    self.popsize += len(newpop)
    newcosts = np.empty(len(newpop)).reshape((len(newpop), 1))
    self.costs = np.vstack([self.costs, newcosts])

    # Recalc costs
    self.cost_me(list(range(self.popsize-len(newpop), self.popsize)))


  def trim_population(self, newsize, tournament = False):
    """ Remove the poorest agents from a population
    TODO: Add tournament selection
    """
    if tournament:
      num_of_tours = self.popsize // self.params['tournament_size']
      to_keep_per_tour = newsize // num_of_tours

    if not tournament: # Simple trim from the end
      self.population = self.population[:newsize, :]
      self.costs = self.costs[:newsize, :]

    self.popsize = newsize

  def cost_me(self, indices = None):
    """Calculate costs for the gentypes in indices
    Reorders the whole population to reflect the new ranking
    """
    if not indices: indices = list(range(self.popsize))
    # Calculate costs
    self.costs[indices] = self.cost_fun(self.population[indices]) #mind column vector

    # Reorder arrays
    sort_index = np.argsort(self.costs.flatten())  # Reverse order
    self.costs = self.costs[sort_index, :]
    self.population = self.population[sort_index,:]


  def mutate(self,indices, mutagen = 'randswap'):
    """
    Performs one of three mutations on a subset
      >randswap: Pick two elmnts at random and trade places
      >pickinsrt: Grap one rndm elemnt and relocate it to a rndm location
      >neighswap: Pick two adjacent elmnts and swap

    Mutations do not ever leave the agent unchanged (to avoid a single agent dominating)
    """
    if not indices: indices = list(range(self.popsize))

    assert mutagen in ['randswap', 'pickinsert', 'neighswap', 'partreverse']
    newpop = self.population[indices, :].copy()

    # swap random columns
    if mutagen == 'randswap':
      a,b = np.random.choice(range(self.gentype_lngth), size=(2,1), replace = False).flatten() # Pick two seperate
      newpop[:,a], newpop[:,b] = self.population[indices,b], self.population[indices,a]

    # pick 'n' insert
    if mutagen == 'pickinsert':
      pck,ins = np.random.choice(range(self.gentype_lngth), size=(2,1), replace = False).flatten() # Pick two seperate
      newindex = list(range(self.gentype_lngth))
      newindex.insert(ins, newindex.pop(pck)) # pop and place
      newpop = self.population[np.ix_(indices,newindex)]

    # swap neighbours
    if mutagen == 'neighswap':
      a = np.random.randint(1, high=self.gentype_lngth, size=1, dtype=int)[0]
      newpop[:,a], newpop[:,a-1] = self.population[indices,a-1], self.population[indices,a]

    #reverse a part of the path
    if mutagen == 'partreverse':
      max_crossover = self.params['max_crossover']
      genlngth = self.gentype_lngth

      start = np.random.randint(0,genlngth-1)
      end = min(genlngth, start + np.random.randint(1,int(genlngth*max_crossover)))
      new_index = list(range(start)) + list(range(start, end))[::-1] + list(range(end, genlngth))

      newpop = self.population[np.ix_(indices,new_index)]


    # roll zeros back to front
    zero_ind =  np.argwhere(newpop[0,:] == 0)[0,0]
    new_index = list(range(zero_ind,self.gentype_lngth)) + list(range(0,zero_ind))
    newpop = newpop[:,new_index]

    return newpop

  def crossover(self, left_inds, right_inds):
    """Crossover a pair of agents A and B
      Move a part of the genotype from B to A
      Remove doublings
    """

    # Define index range to cross from right to left
    max_crossover = self.params['max_crossover']
    genlngth = self.gentype_lngth

    start = np.random.randint(1,genlngth-1)
    end = min(genlngth, start + np.random.randint(1,int(genlngth*max_crossover)))
    cross_index = list(range(start, end))

    # Grab part from right, make negative as label
    cross_part = -self.population[np.ix_(right_inds, cross_index)]

    # Insert into left
    full_list = np.hstack([self.population[left_inds, :start], cross_part, self.population[left_inds, start:]])
    newpop = np.empty(self.population[left_inds,:].shape, dtype = int)
    # Remove positive matches
    for i, ind in enumerate(left_inds):
      # Keep all items not in cross_part
      newpop[i,:]= full_list[i, np.logical_not(np.isin(full_list[i,:], -cross_part[i, :]))]
    newpop = np.absolute(newpop)

    return newpop


  def plot_cost_profile(self, ax):
    ax.hist(self.costs, bins = 50)
    # ax.set_yscale('log')
    # ax.set_xscale('log')





#
# Profiler set up
#


def profile_setup():
  n_prof = 500000
  k_prof = 32
  N_prof = NodeList(k_prof, .45)
  G_prof = GeneticOptimiser(gentype_lngth = k_prof, popsize = n_prof, cost_fun = N_prof.path_length)
  return G_prof

def profile_me(G_prof):
  for i in range(50):
    G_prof.evolve()





#
# MAIN set-up
#

if __name__ == '__main__':


  k = 32  # Node count
  n = 5000   # Population size
  evo_steps = 50
  num_of_reports = 10


  N = NodeList(k, .45)

  G = GeneticOptimiser(gentype_lngth = k, popsize = n, cost_fun = N.path_length)


  for stp in range(evo_steps+1):

    if stp % (evo_steps // num_of_reports) == 0:

      print('{}: {}'.format(stp, G.costs[0]))

      f2, [ax1, ax2] = plt.subplots(1,2)
      N.plot(ax2, G.population[0])
      ax2.set_xlim(-1.6, 1.6)
      ax2.set_ylim(-1.6, 1.6)
      ax2.set_aspect(1)
      G.plot_cost_profile(ax1)
      f2.savefig('{:4d}.jpg'.format(stp))
      plt.axes().set_aspect('equal', 'datalim')
      plt.close()

    G.evolve()

