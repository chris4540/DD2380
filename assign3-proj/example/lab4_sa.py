
# coding: utf-8

# #  Lab 4 -  Simulated Annealing
#
# ##### Keywords: simulated annealing, metropolis-hastings, MCMC, temperature, travelling salesman problem

# ## Contents
# {:.no_toc}
# *
# {: toc}

# In[128]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

## for TSP material
import random
import time
import itertools
import urllib
import csv
import functools
from statistics import mean, stdev
import math


# ## Optimization
#
# Welcome to Lab 4.  Today we're going to discuss simulated annealing (SA) with a particular focus on the Traveling Salesman Problem frequently referred to as TSP.  We introduced simulated annealing in the last lecture, but I'll briefly discuss some of the salient points here trying to give a slightly different viewpoint.
#
# We often come across a problem where given a function defined on a domain $D$ we wish to find a member of $D$ for which that function is minimized (or maximized).  This is known as **optimization**.  There are many optimization algorithms and your selection of a particular optimization scheme may be dependent on characteristics such as the number of variables in your function, whether it is continuous, whether you wish guaranteed convergence, whether a global optima is required as opposed to a local optima, etc.
#
# We've already seen one optimization algorithm **gradient descent**.  That algorithm is appropriate for functions with a gradient defined everywhere and for which local minima are sufficient.  In the future we may see other optimization algorithms like **genetic algorithms** or **expectation maximization**.
#
# **Simulated Annealing** is a very populare optimization algorithm because it's very robust to different types of functions (e.g. no continuity, differentiability  or dimensionality requirements) and can find global minima/maxima.

# ## Motivating Simulated Annealing
#
# ### Sampling from our  function
#
# In any optimization problem we're given a function for which we want to find a minimum or maximum value.  The most basic way to go about finding that optimum point is to sample from the function domain and keep track of the values at each of the points we've visited.  How do we sample from the function though?
#
# We have a few tools in our arsenal so far including **inverse transform sampling** and variants of **rejection sampling**. Our problem is that all of these methods sample globally from our function.  Since the domain of the functions in most optimization functions is absurdly large if not infinite, it doesn't make sense to sample and keep track of all the values you've visited because that's an intractable approach.  You really want to find one of the relatively few troughs (or peaks) of your function and once you've found them move with some sense of inevitability to the optimum of that region.  In order for such a scheme to work you need to have some ability to do *contextually dependent sampling*  where your next sample has some dependence on the current sample but still preserving the representational relationship of your samples (in aggregate) to your function.  **Markov Chain Monte Carlo (MCMC)** sampling methods and in particular **Metropolis-Hastings (MH)** gives us such a sampling mechanism.  We'll discuss and prove these properties of MH in the future.  For now, we'll present the MH algorithm without much additional explanation.
#
# https://image.slidesharecdn.com/mcmc-12644140116474-phpapp02/95/markov-chain-monte-carlo-explained-10-728.jpg?cb=1264392537

# ![Metropolis-Hastings algorithm](https://image.slidesharecdn.com/mcmc-12644140116474-phpapp02/95/markov-chain-monte-carlo-explained-10-728.jpg?cb=1264392537)
#
# (via [MCMC Explained](http://www.slideshare.net/dariodigiuni/markov-chain-monte-carlo-explained))

# ### What about temperature?
#
# Given a function $p(x)$ we can rewrite that function in following way:
#
# $$p(x) = e^{-(-\log(p(x))}$$
#
# So if define the energy density for a function as $E(x)\equiv-\log{p(x)}$
#
# We can now aim to sample from the function parameratized by a Temperature $T$.
#
# $$p(x\vert T) = e^{-\frac{1}{T} E(x)} = p(x)^{\frac{1}{T}}$$
#
# For practical purposes we generally consider $T \in [0,1]$ which has the effect of exagerating the peaks of the original function.  When $T = 1$ we're sampling from the orignial function.  Let's use a 2-dimensional example to observe what happens at different temperatures for a sample function.
#
#

# In[9]:


# Let's plot a target distribution p(x,y)

ee=0.01
p = lambda x,y: np.exp(-(x-y)**2 / (2*ee) - (x+y)**2/2) + np.exp( -(x+.30)**2/0.01)*np.exp( -(y-.30)**2/0.01)

xx= np.linspace(-1,1,100)
yy= np.linspace(-1,1,100)

M = np.empty((100,100))
i=0
for x in xx:
    j=0
    for y in yy:
        M[j,i]=p(x,y)
        j +=1
    i +=1


plt.contour(xx,yy,M)
plt.colorbar()
plt.show()


# Now let's look at how our Metropolis-Hastings algorithm travels this function/distribution different temperatures.

# In[10]:


# METROPOLIS ALGORITHM
#create a simple Metropolis Hastings function
def MH_simple(p, n, sig, T, x0, y0):
    x_prev = x0
    y_prev = y0
    x=[x_prev]; y=[y_prev]

    k=1
    i=1

    while i<n:

        x_star = np.random.normal(x_prev, sig)
        y_star = np.random.normal(y_prev, sig)

        P_star = p(x_star, y_star)
        P_prev = p(x_prev, y_prev)

        U =  np.random.uniform()

        E_star = -np.log(P_star)
        E_prev = -np.log(P_prev)

        A =  np.exp(-(1.0/T) * (E_star-E_prev))
        if U < A:
            x.append(x_star); y.append(y_star);
            i = i + 1
            x_prev = x_star; y_prev = y_star
        else :
            x.append(x_prev); y.append(y_prev);
            #x_prev = x[i] ; y_prev = y[i]
            i = i + 1

            k=k+1
    return np.array(x),np.array(y), k


# In[11]:


# number of samples
n = 2000
# step size
sig =.05

#intitialize the sampling. Start somewhere from -1..1
x_start = np.random.uniform(low=-1, high=1)
y_start = np.random.uniform(low=-1, high=1)


# In[31]:


xSL, ySL, SLrejected = MH_simple(p, n, sig=sig, T=0.01, x0=x_start, y0=y_start)
print "Number of rejected samples at super low temperature: "
print SLrejected

xL, yL, Lrejected = MH_simple(p, n, sig=sig, T=0.1, x0=x_start, y0=y_start)
print "Number of rejected samples at low temperature: "
print Lrejected

xM, yM, Mrejected = MH_simple(p, n, sig=sig, T=0.7, x0=x_start, y0=y_start)
print "Number of rejected samples at medium temperature: "
print Mrejected

xH, yH, Hrejected = MH_simple(p, n, sig=sig, T=1.0, x0=x_start, y0=y_start)
print "Number of rejected samples at high temperature: "
print Hrejected


# In[32]:


plt.subplot(2,2,1)
plt.contour(xx,yy,M, alpha=0.4)
plt.plot(xSL,ySL, 'y.-', alpha=0.3)
plt.plot(xSL[-1],ySL[-1], 'rs', ms=10)

plt.subplot(2,2,2)
plt.contour(xx,yy,M, alpha=0.4)
plt.plot(xL,yL, 'b.-', alpha=0.3)
plt.plot(xL[-1],yL[-1], 'rs', ms=10)

plt.subplot(2,2,3)
plt.contour(xx,yy,M, alpha=0.4)
plt.plot(xM,yM, 'g.-', alpha=0.3)
plt.plot(xM[-1],yM[-1], 'ks', ms=10)

plt.subplot(2,2,4)
plt.contour(xx,yy,M, alpha=0.4)
plt.plot(xH,yH, 'r.-', alpha=0.3)
plt.plot(xH[-1],yH[-1], 'rs', ms=10)


# ## The Simulated Annealing Algorithm
#
# So now we have a better sense of how to find peaks (valleys) and then find optima.  First use Metropolis-Hastings sampling at high temperatures to travel around the distribution and find all the peaks (valleys).  Slowly cool down the temperature which will focus you in a particular optima region and allow you to find the optima.
#
# 1. Initialize $x_i,T, L(T)$ where $L$ is the number of iterations at a particular temperature.
# 2. Perform $L$ transitions thus(we will call this an epoch):
#     1. Generate a new proposed position $x$
#     2. If $x$ is accepted (according to probability $P = e^{(-\Delta E/T)}$, set $x_{i+1} = x_j$, else set $x_{i+1} = x_{i}$
# 3. Update T and L
# 4. Until some fixed number of epochs, or until some stop criterion is fulfilled, goto 2.
#

# ![SA flowchart](http://apmonitor.com/me575/uploads/Main/sim_annealing_flow.png)
#
# (via [BYU ME 575 SA Tutorial](http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing))

# ## Example from Lecture
#
# Let us consider the example of a function with one global minimum and many local minima.

# In[4]:


f = lambda x: x**2 + 4*np.sin(2*x)


# [Wolfram Alpha](http://bit.ly/2lTPKsB) is a useful way to get info on this function, but here is what it looks like...

# In[66]:


xs = np.linspace(-10.,10.,1000)
plt.plot(xs, f(xs));


# ##  Cooling Schedule and convergence
#
# The sequence of temperatures we use (we'll call the iterations for each temperature an **epoch**), and the number of iterations at each temperature, called the stage length, constitute the cooling schedule.
#
# Why do we think we will reach a global minimum? For this we'll have to understand the structure of the sequence of positions simulated annealing gives us. We'll see this in more detail later, but, briefly, simulated annealing produces either a set of homogeneous markov chains, one at each temperature or a single inhomogeneous markov chain.
#
# Now, assume that our proposal is symmetric: that is proposing $x_{i+1}$ from $x_i$ in $\cal{N}(x_i)$ has the same probability as  proposing $x_{i}$ from $x_{i+1}$ in $\cal{N}(x_{i+1})$. This *detailed balance* condition ensures that the sequence of $\{x_t\}$ generated by simulated annealing is a stationary markov chain with the boltzmann distribution as the stationary distribution of the chain as $t \to \infty$. Or, in physics words, you are in equilibrium (with detailed balance corresponding to the reversibility of the isothermal process).
#
# Then the appropriately normalized Boltzmann distribution looks like this (assuming M global minima (set $\cal{M}$) with function value $f_{min}$):
#
# $$ p(x_i) = \frac{e^{-(f(x_i) - f_{min})/T}}{M + \sum_{j \notin  \cal{M}} e^{-(f(x_i) - f_{min})/T}}$$
#
# Now as $T \to 0$ from above, this becomes $1/M$ if $x_i \in \cal{M}$ and 0 otherwise.
#
# We will talk more about markov chains soon, they are at the root of MCMC which is very similar to  simulated annealing.

# In[371]:


import functools
distx = lambda g, x: np.e**(-g(x))
dxf = functools.partial(distx, f)
outx = np.linspace(-10, 10,1000)
import scipy.integrate as integrate
O=20
plt.plot(outx, dxf(outx)/integrate.quad(dxf,-O, O)[0]);
A=integrate.quad(lambda x: dxf(x)**1.2,-O, O)[0]
plt.plot(outx, (dxf(outx)**1.2)/A);
B=integrate.quad(lambda x: dxf(x)**2.4,-O, O)[0]
plt.plot(outx, (dxf(outx)**2.4)/B);
C=integrate.quad(lambda x: dxf(x)**10,-O, O)[0]
plt.plot(outx, (dxf(outx)**10)/C);


plt.xlim([-5,5])
# plt.xscale('log')
# plt.yscale('log')


# ![xkcd travelling salesman](http://imgs.xkcd.com/comics/travelling_salesman_problem.png)
#
# (via [xkcd](http://xkcd.com/399/))
#
#

# ### Proposal
#
# Let us first define our proposal distribution as a normal centers about our current position. In this case we need to figure a width. This is our first case of tuning, the **width** of our neighborhood $\cal{N}$.  Notice that this is a symmetric proposal and will thus follow the detailed balance condition.

# In[215]:


pfxs = lambda s, x: x + s*np.random.normal()
pfxs(0.1, 10)


# Too wide a width and we will lose sensitivity near the minimum..too narrow and we'll spend a very long time heading downward on the function.
#
# We capture the width since we are not planning to write an adaptive algorithm (we must learn to walk before we can run...)

# In[216]:


from functools import partial
pf = partial(pfxs, 0.1)
pf(10)


# ### Cooling Schedule
#
# Now we must define the functions that constitute our cooling schedule. We reduce the temperature by a multiplicative factor of 0.8, and increase the epoch length by a factor of 1.2

# In[307]:


import math
tf = lambda t: 0.8*t #temperature function
itf = lambda length: math.ceil(1.2*length) #iteration function


# ### Running the Algorithm
#
# We define the `sa` function that takes a set of initial conditions, on temperature, length of epoch, and starting point, the energy function, the number or opochs to run, the cooling schedule, and the proposal, and implements the algorithm defined above. Our algorithms structure is that of running for some epochs during which we reduce the temperature and increase the epoch iteration length. This is somewhat wasteful, but simpler to code, although it is not too complex to build in stopping conditions.

# In[308]:


def sa(energyfunc, initials, epochs, tempfunc, iterfunc, proposalfunc):
    accumulator=[]
    best_solution = old_solution = initials['solution']
    T=initials['T']
    length=initials['length']
    best_energy = old_energy = energyfunc(old_solution)
    accepted=0
    total=0
    for index in range(epochs):
        print("Epoch", index)
        if index > 0:
            T = tempfunc(T)
            length=iterfunc(length)
        print("Temperature", T, "Length", length)
        for it in range(length):
            total+=1
            new_solution = proposalfunc(old_solution)
            new_energy = energyfunc(new_solution)
            # Use a min here as you could get a "probability" > 1
            alpha = min(1, np.exp((old_energy - new_energy)/T))
            if ((new_energy < old_energy) or (np.random.uniform() < alpha)):
                # Accept proposed solution
                accepted+=1
                accumulator.append((T, new_solution, new_energy))
                if new_energy < best_energy:
                    # Replace previous best with this one
                    best_energy = new_energy
                    best_solution = new_solution
                    best_index=total
                    best_temp=T
                old_energy = new_energy
                old_solution = new_solution
            else:
                # Keep the old stuff
                accumulator.append((T, old_solution, old_energy))

    best_meta=dict(index=best_index, temp=best_temp)
    print("frac accepted", accepted/total, "total iterations", total, 'bmeta', best_meta)
    return best_meta, best_solution, best_energy, accumulator


# Lets run the algorithm

# In[310]:


inits=dict(solution=8, length=100, T=100)
bmeta, bs, be, out = sa(f, inits, 30, tf, itf, pf)


# Our global minimum is as predicted, and we can plot the process:

# In[311]:


bs, be


# In[325]:


xs = np.linspace(-10.,10.,1000)
plt.plot(xs, f(xs), lw=2);
eout=list(enumerate(out))
#len([e[1] for i,e in eout if i%100==0]), len([e[2] for i,e in eout if i%100==0])
plt.plot([e[1] for i,e in eout if i%100==0], [e[2] for i,e in eout if i%100==0], 'o-', alpha=0.6, markersize=5)
#plt.plot([e[1] for e in out], [f(e[1]) for e in out], '.', alpha=0.005)
plt.xlim([-3,10])
plt.ylim([-10,100])


# Notice how the x-values start forming a tighter and tighter (stationary) distribution about the minimum as temperature is decreased.

# In[312]:


plt.plot([e[0] for e in out],[e[1] for e in out],'.', alpha=0.1, markersize=5);
plt.xscale('log')
plt.xlabel('T')
plt.ylabel('x')


# ...which translates to a tight distribution on the "energy" $f$.

# In[314]:


plt.plot([e[0] for e in out],[e[2] for e in out],'.', alpha=0.1, markersize=5);
plt.xscale('log')
plt.xlabel('T')
plt.ylabel('f')


# We can also plot these against the iterations.

# In[316]:


plt.plot([e[1] for e in out], alpha=0.2, lw=1);
plt.xscale('log')
plt.xlabel('iteration')
plt.ylabel('x')


# In[317]:


plt.plot(range(len(out)),[e[2] for e in out], alpha=0.2, lw=1);
plt.xscale('log')
plt.xlabel('iteration')
plt.ylabel('f')


# We can make an animation of the process of finding a global minimum.

# In[326]:


def animator(f, xvals, xdatafn, ydatafn, frameskip, ax):

    fvals=[f(x) for x in xvals]
    ax.plot(xvals, fvals, lw=3)
    line, = ax.plot([], [], "o", markersize=12)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x_n = xdatafn(i, frameskip)
        y_n = ydatafn(i)
        line.set_data(x_n, y_n)
        return line,
    return init, animate


# In[328]:


import matplotlib.animation as animation
from JSAnimation import IPython_display
fig = plt.figure()
ax = plt.axes()
fskip=1000
sols=[e[1] for e in out]
smallsols = sols[0:bmeta['index']:fskip]+sols[bmeta['index']::fskip//2]
xdatafn = lambda i, fskip: smallsols[i]
ydatafn = lambda i: f(xdatafn(i, fskip))
i, a = animator(f, xs, xdatafn, ydatafn, fskip, ax)
anim = animation.FuncAnimation(fig, a, init_func=i,
                        frames=len(smallsols), interval=300)
anim.save('images/sa1.mp4')
anim


# ## Traveling Salesman Problem (TSP)
#
# This section and the ones that follow is heavily indebted to (if not lifted almost entirely from) Peter Norvig's iPython Notebook on the TSP.  I highly recommend you visit that notebook and all his other notebooks at http://norvig.com/ipython/
#
# Some optimization problems are ***combinatorial***, in the sense that there are $p$ items that can be ordered or combined in many different ways, some ways being better than others according to a set of specified criteria.
#
# Consider the [*Traveling Salesperson Problem*](http://en.wikipedia.org/wiki/Traveling_salesman_problem):
#
# > *Given a set of cities and the distances between each pair of cities, what is the shortest possible tour that visits each city exactly once, and returns to the starting city?*
#
# Assuming travel is the same distance irrespective of travel direction, there are:
#
# $$\frac{(p-1)!}{2}$$
#
# possible routes. So, 5 cities have 120 possible routes, 10 cities have 181,440 routes, 50 cities have $3 \times 10^{64}$ routes!
#
# <a href="http://www.math.uwaterloo.ca/tsp/history/pictorial/dfj.html"><img src="http://www.math.uwaterloo.ca/tsp/history/img/dantzig.gif"></a>
# <center>An example tour.</center>
#
# ### Developing some Vocabulary
#
# Do we understand precisely what the problem is asking? Do we understand all the concepts that the problem talks about?  Do we understand them well enough to implement them in a programming language? Let's take a first pass:
#
# - **A set of cities**: We will need to represent a set of cities; Python's `set` datatype might be appropriate.
# - **Distance between each pair of cities**: If `A` and `B` are cities, this could be a function, `distance(A, B),` or a table lookup, `distance[A][B]`.  The resulting distance will be a real number.
# - **City**: All we have to know about an individual city is how far it is from other cities. We don't have to know its name, population, best restaurants, or anything else. So a city could be just an integer (0, 1, 2, ...) used as an index into a distance table, or a city could be a pair of (x, y) coordinates, if we are using straight-line distance on a plane.
# - **Tour**: A tour is a specified order in which to visit the cities; Python's `list` or `tuple` datatypes would work. For example, given the set of cities `{A, B, C, D}`, a tour might be the list `[B, D, A, C]`, which means to travel from `B` to `D` to `A` to `C` and finally back to `B`.
# - **Shortest possible tour**: The shortest tour is the one whose tour length is the minimum of all tours.
# - **Tour length**: The sum of the distances between adjacent cities in the tour (including the last city back to the first city).   Probably  a function, `tour_length(tour)`.
#
# Let's start with a bruteforce algorithm that is guaranteed to solve the problem, although it is inefficient for large sets of cities:
#
# > **All Tours Algorithm**: *Generate all possible tours of the cities, and choose the shortest tour (the one with minimum tour length).*
#
#
# Let's begin the implementation:

# In[34]:


def alltours_tsp(cities):
    "Generate all possible tours of the cities and choose the shortest tour."
    return shortest_tour(alltours(cities))

def shortest_tour(tours):
    "Choose the tour with the minimum tour length."
    return min(tours, key=tour_length)


# **Note**: In Python `min(`*collection*`,key=`*function*`)` means to find the element *x* that is a member of *collection* such that *function(x)* is minimized. So `shortest` finds the tour whose `tour_length` in the minimal among the tours.
#
# Representing Tours
# ------------------
#
# A tour starts in one city, and then visits each of the other cities in order, before returning to the start city.  A natural representation of a tour is a sequence of cities. For example `(1, 2, 3)` could represent a tour that starts in city 1, moves to 2, then 3, and finally returns to 1.
#
#
# Now for the `alltours` function.  If a tour is a sequence of cities, then all the tours are *permutations* of the set of all cities. A function to generate all permutations of a set is already provided in Python's standard `itertools` library module; we can use it as our implementation of `alltours`.  We take some steps to make tours non-redundant as well.

# In[36]:


def alltours(cities):
    "Return a list of tours, each a permutation of cities, but each one starting with the same city."
    start = first(cities)
    return [[start] + Tour(rest)
            for rest in itertools.permutations(cities - {start})]

def first(collection):
    "Start iterating over collection, and return the first element."
    return next(iter(collection))

Tour = list  # Tours are implemented as lists of cities


# The length of a tour is the sum of the lengths of each edge in the tour; in other words, the sum of the distances between consecutive cities in the tour, including the distance form the last city back to the first:

# In[37]:


def tour_length(tour):
    "The total of distances between each pair of consecutive cities in the tour."
    return sum(distance(tour[i], tour[i-1])
               for i in range(len(tour)))


# **Note**: We use one Python-specific trick: when `i` is 0, then `distance(tour[0], tour[-1])` gives us the wrap-around distance between the first and last cities, because `tour[-1]` is the last element of `tour`.
#
# Representing Cities
# --------------------------------
#
# We determined that the only thing that matters about cities is the distance between them. But before we can decide about how to represent cities, and before we can define `distance(A, B)`,  we have to make a choice. In the fully general version of the TSP, the "distance" between two cities could be anything: it could factor in the amount of time it takes to travel between cities, the twistiness of the road, or anything else. The `distance(A, B)` might be different from `distance(B, A)`. So the distances could be represented by a matrix `distance[A][B]`, where any entry in the matrix could be any (non-negative) numeric value.
#
# But we will ignore the fully general TSP and concentrate on an important special case, the **Euclidean TSP**, where the distance between any two cities is the [Euclidean distance](http://en.wikipedia.org/wiki/Euclidean_distance), the straight-line distance between points in a two-dimensional plane. So a city can be represented by a two-dimensional point: a pair of *x* and *y* coordinates. We will use the constructor function `City`, so that `City(300, 0)` creates a city with x-coordinate of 300 and y coordinate of 0.  Then `distance(A, B)` will be a function that uses the *x* and *y* coordinates to compute the distance between `A` and `B`.
#
# Representing Points and Computing `distance`
# ---
#
# OK, so a city can be represented as just a two-dimensional point. But how will we represent points?  Here are some choices, with their pros and cons:
#
# * **tuple:** A point is a two-tuple of (*x*, *y*) coordinates, for example, `(300, 0)`. **Pro:** Very simple.
# **Con:** doesn't distinguish Points from other two-tuples.
#
# * **class:** Define a custom `Point` class with *x* and *y* slots. **Pro:** explicit, gives us `p.x` and `p.y` accessors.  **Con:** less efficient.
#
# * **complex:** Python already has the two-dimensional point as a built-in numeric data type, but in a non-obvious way: as `complex` numbers, which inhabit the two-dimensional (real &times; imaginary) plane.  **Pro:** efficient. **Con:** a little confusing; doesn't distinguish Points from other complex numbers.
# * **subclass of complex:** All the pros of `complex`, and eliminating the major con.
#
#
# Any of these choices would work perfectly well; We decided to use a subclass of `complex`:

# In[47]:


# Cities are represented as Points, which are a subclass of complex numbers

class Point(complex):
    x = property(lambda p: p.real)
    y = property(lambda p: p.imag)

City = Point

def distance(A, B):
    "The distance between two points."
    return abs(A - B)

def Cities(n, width=900, height=600, seed=42):
    "Make a set of n cities, each with random coordinates within a (width x height) rectangle."
    random.seed(seed * n)
    return frozenset(City(random.randrange(width), random.randrange(height))
                     for c in range(n))

def plot_tour(tour):
    "Plot the cities as circles and the tour as lines between them. Start city is red square."
    start = tour[0]
    plot_lines(list(tour) + [start])
    plot_lines([start], 'rs') # Mark the start city with a red square

def plot_lines(points, style='bo-'):
    "Plot lines to connect a series of points."
    plt.plot([p.x for p in points], [p.y for p in points], style)
    plt.axis('scaled'); plt.axis('off')

def plot_tsp(algorithm, cities):
    "Apply a TSP algorithm to cities, plot the resulting tour, and print information."
    # Find the solution and time how long it takes
    t0 = time.clock()
    tour = algorithm(cities)
    t1 = time.clock()
    assert valid_tour(tour, cities)
    plot_tour(tour); plt.show()
    print("{} city tour with length {:.1f} in {:.3f} secs for {}"
          .format(len(tour), tour_length(tour), t1 - t0, algorithm.__name__))

def valid_tour(tour, cities):
    "Is tour a valid tour for these cities?"
    return set(tour) == set(cities) and len(tour) == len(cities)


# ## Heuristics
#
# Unfortunately, there is no known combinatorial optimization algorithm for obtaining an optimal solution to the TSP in polynomial time. Instead, we must turn to ***heuristics***, which have no guarantee of a global maximum, but in practice tend to yield *good* results in a reasonable time. Thus, we are trading off global optimality for a little speed.
#
# Heuristics have two notable characteristics:
#
# * **iteration**: candidate solutions are incrementally improved
# * **localization**: search for improved solutions are restricted to a local neighborhood of the current solution
#
# This ***local search*** approach encompasses several specific techniques, some of which we will explore here. For a given candidate solution vector $\mathbf{\theta}^{(t)}$ at iteration $t$, we might change components $\theta_i$ to propose an updated solution $\mathbf{\theta}^{(t+1)}$. Limiting the number of changes keeps $\mathbf{\theta}^{(t+1)}$ in the *neighborhood* of $\mathbf{\theta}^{(t)}$. We refer to $k$ changes to the candidate solution as a **k-change** and the set of possible new candidates as the *k-neighborhood*.
#
# A sensible approach for updating a candidate solution is to choose the best candidate from the neighborhood; this is called ***steepest ascent***. The selection of any improved candidate is called an *ascent*. However, choosing the steepest ascent from a neighborhood may not be globally optimal if, for example, it takes us toward a local maximum at the cost of missing a global maximum. An algorithm that uses a steepest ascent strategy in the context of local search is called a *greedy* algorithm.
#
# We can construct pretty readily a greedy algorithm for TSP.
#
# Here is a description of the nearest neighbor algorithm:
#
# > **Nearest Neighbor Algorithm:** *Start at any city; at each step extend the tour by moving from the previous city to its nearest neighbor that has not yet been visited.*
#
# So now, instead of considering all *n*! tours, we are generating a single tour. It takes O(*n*<sup>2</sup> ) time to find the tour, because it has *n*-1 steps, and at each step we consider each of the remaining cities.
# I implement the algorithm as follows:
#
# * "*Start at any city*": arbitrarily pick the first city.
# * "*extend the tour*": append to the end of a list of cities.
# * "*by moving from the previous city*": previous city is `tour[-1]`.
# * "*to its nearest neighbor*": define the function `nearest_neighbor`.
# * "*that has not yet been visited*": keep a set of `unvisited` cities.
#
# That gives us:

# In[42]:


def nn_tsp(cities):
    """Start the tour at the first city; at each step extend the tour
    by moving from the previous city to the nearest neighboring city, C,
    that has not yet been visited."""
    start = first(cities)
    tour = [start]
    unvisited = set(cities - {start})
    while unvisited:
        C = nearest_neighbor(tour[-1], unvisited)
        tour.append(C)
        unvisited.remove(C)
    return tour

def nearest_neighbor(A, cities):
    "Find the city in cities that is nearest to city A."
    return min(cities, key=lambda c: distance(c, A))


# In[49]:


plot_tsp(alltours_tsp, Cities(10, seed=332))


# In[48]:


plot_tsp(nn_tsp, Cities(10, seed=332))


# In order to attain a global maximum (or globally-competitive solution), it makes sense for a heuristic approach to  occasionaly choose an candidate solution that is not the best-in-neighborhood. In other words, to move from one peak to another (higher) peak, one must pass through valleys.  Hmmmm.... Where can we find an approach that does this?  Oh yeah!! Our good old buddy **simulated annealing**.
#

# In[129]:




def reverse_segment(input_tour, i, j):
    "Reverse segment tour[i:j] of a tour"
    input_tour[i:j] = reversed(input_tour[i:j])

def swap_cities(input_tour, i, j):
    "Swap two cities at index i and j in a tour"

    # save city1
    city1 = input_tour[i]

    # save city2
    city2 = input_tour[j]

    new_tour = input_tour[:]

    # swap
    new_tour[j] = city1
    new_tour[i] = city2

    return new_tour



def change_tour(input_tour):
    "Change a tour for tsp iteration"

    indices = range(len(input_tour))

    # take two random indices to swap
    c1 = np.random.choice(indices)
    c2 = np.random.choice(indices)

    new_tour = change_path(input_tour, c1, c2)

    return new_tour



change_path=swap_cities







# We've constructed our own simulated annealing function for tsp but we don't
# really need to make any changes.  So we'll just comment the regular sa :-)
def sa_tsp(energyfunc, initials, epochs, tempfunc, iterfunc, proposalfunc):
    """Run simulated annealing on a tsp."""

    # Accumulate results in the same form as initals
    accumulator=[]

    # Our initial state is in initials['solution']
    best_solution = old_solution = initials['solution']

    # Our initial temperature is in initials['T']
    T=initials['T']

    # Our initial length (i.e. number of iterations per epoch)
    # is in initals['length']
    length=initials['length']

    # initialize the energy of our current state by running the
    # energy function on our initial solution
    best_energy = old_energy = energyfunc(old_solution)

    # keep track of accepted proposals and total iterations
    accepted=0
    total=0

    for index in range(epochs):
        #print("Epoch", index)

        # if we're past the first index, we need
        # to update our cooling schedule and iteration
        # schedule
        if index > 0:
            T = tempfunc(T)
            length=iterfunc(length)

        #print("Temperature", T, "Length", length)

        # run through the iterations for each epoch
        for it in range(length):

            # keep track of total proposals
            total+=1

            # get a new proposal and calculate its energy
            new_solution = proposalfunc(old_solution)
            new_energy = energyfunc(new_solution)

            # Use a min here as you could get a "probability" > 1
            alpha = min(1, np.exp((old_energy - new_energy)/T))
            if ((new_energy < old_energy) or (np.random.uniform() < alpha)):

                # Accept proposed solution
                accepted+=1.0
                accumulator.append((T, new_solution, new_energy))

                # we have a new candidate for optimum (minimum)
                if new_energy < best_energy:
                    # Replace previous best with this one
                    best_energy = new_energy
                    best_solution = new_solution
                    best_index=total
                    best_temp=T

                old_energy = new_energy
                old_solution = new_solution
            else:
                # Keep the old stuff
                accumulator.append((T, old_solution, old_energy))

    best_meta=dict(index=best_index, temp=best_temp)
    print("frac accepted", accepted/total, "total iterations", total, 'bmeta', best_meta)
    return best_meta, best_solution, best_energy, accumulator


# In[135]:


initial_cities = Cities(25)
initial_tour = list(initial_cities)
length_func1 = lambda temperature: np.max((np.floor(np.sqrt(temperature)).astype(int),1))
length_func2 = lambda length: max(int(math.ceil(1.2*length)), 10)
length_func = length_func1
temp_func = lambda t: 0.8*t

init_length = length_func(100)

inits=dict(solution=initial_tour, length=init_length, T=3.0)

print inits
bmeta, bs, be, out = sa_tsp(tour_length, inits, 10000, temp_func, length_func, change_tour);


# In[136]:


nn_sol = nn_tsp(initial_cities)
plot_tour(nn_sol)


# In[137]:


print be
plot_tour(bs)


# In[138]:


plt.plot(range(len(out)),[e[2] for e in out], alpha=0.6, lw=1);
plt.xscale('log')
plt.xlabel('iteration')
plt.ylabel('f')


# ## Practical Choices
#
# *  Start with temperature $T_0$ large enough to accept all transitions. Its working against $\Delta E$, so should be at-least a bit higher than it.
# * The proposal step size should lead to states with similar energy. This would exclude many bad steps
# *  Lowering temperature schedule (thermostat). Common choices are
#     1. Linear: Temperature decreases as  $T_{k+1} = \alpha T_k$. Typical values are $0.8 < \alpha < 0.99$. $k$ indexes epochs.
#     2. Exponential: Temperature decreases as $0.95^{{\rm k}}$
#     3. Logarithmic: Temperature decreases as $1/\log({\rm k})$
# * Reannealing interval, or epoch length is the number of points to accept before reannealing (change the temperature). Typical starting value is 100, and you want to increase it as  $L_{k+1} = \beta L_k$ where $\beta > 1$.
# * Larger decreases in temperature require correspondingly longer epoch lengths to re-equilibriate
# * Running long epochs at larger temperatures is not very useful. In most problems local minima can be jumped over even at low temperatures. Thus it may be useful to decrease temperature rapidly at first.
# * Stopping criterion
#     1. Max iterations bounds the number of iterations the algorithm takes
#     2. Function tolerance. The algorithm stops if the average change in the objective function after  $m$ iterations is below user specified tolerance
#     3. Objective limit. The algorithm stops if the objective function goes below some value
#
# - It can be shown (although this is too slow a bound) that convergence to a set of global maxima is assured for $T_i = 1/(Clog(i + T_0))$ where $C$ and $T_0$ are problem dependent. The usual interpretation is that $C$ is the height of the tallest local minimum.
# - Sometimes reheating is useful to explore new areas.

# ## Sudoku
#
# For those who haven't seen these before, a sudoku puzzle typically consists of a square grid of 3x3 blocks each containing 3x3 cells. This has of course be scaled up to n>3 for really sadistic gamers, but we're going to stick with the well-known 3x3 situtation. The puzzle begins with  a variable number of  integers in specified (fixed) positions on the game board. The objective is to fill in the remaining cells so that each row, column and block contains only distinct and unique integers from 1 to 9. Easy! (...or not). Depending on the start position, these can sometimes be solved systematically by selecting individual empty cells that can only have one possible outcome based on the misisng integers from the associated row, column and block.
#
# Here is an example where a systematic approach will not work:

# In[139]:


def plot_sudoku(n):
    # Simple plotting statement that ingests a 9x9 array (n), and plots a sudoku-style grid around it.
    plt.figure()
    for y in range(10):
        plt.plot([-0.05,9.05],[y,y],color='black',linewidth=1)

    for y in range(0,10,3):
        plt.plot([-0.05,9.05],[y,y],color='black',linewidth=3)

    for x in range(10):
        plt.plot([x,x],[-0.05,9.05],color='black',linewidth=1)

    for x in range(0,10,3):
        plt.plot([x,x],[-0.05,9.05],color='black',linewidth=3)

    plt.axis('image')
    plt.axis('off') # drop the axes, they're not important here

    for x in range(9):
        for y in range(9):
            foo=n[8-y][x] # need to reverse the y-direction for plotting
            if foo > 0: # ignore the zeros
                T=str(foo)
                plt.text(x+0.3,y+0.2,T,fontsize=20)

    plt.show()


# In[140]:


sudoku_hard=np.array([[0,0,0,9,0,0,7,2,8],
                      [2,7,8,0,0,3,0,1,0],
                      [0,9,0,0,0,0,6,4,0],
                      [0,5,0,0,6,0,2,0,0],
                      [0,0,6,0,0,0,3,0,0],
                      [0,1,0,0,5,0,0,0,0],
                      [1,0,0,7,0,6,0,3,4],
                      [0,0,0,5,0,4,0,0,0],
                      [7,0,9,1,0,0,8,0,5]])
plot_sudoku(sudoku_hard)

#sudoku_easy=np.array([[0,0,0,0,0,0,1,0,0],
#                      [0,0,0,7,0,0,0,9,4],
#                      [4,0,0,1,0,0,2,0,0],
#                      [0,2,0,8,0,0,3,5,0],
#                      [5,9,0,0,0,0,0,2,6],
#                      [0,6,8,0,0,5,0,7,0],
#                      [0,0,9,0,0,6,0,0,2],
#                      [8,1,0,0,0,3,0,0,0],
#                      [0,0,5,0,0,0,0,0,0]])
#plot_sudoku(sudoku_easy)


# Does brute force work any better here than TSP?
# <b>Answer:</b> let's just do a simple order of magnitude, back of the envelope style calculation.<br>
#
# There are $9 \times 9 =81$ squares in our grid. 26 of these are aleady filled. Leaving 65 empty squares. There are 9 possible numbers that can go into each of the 65 squares, so there are $\sim 10^{60}$ combinations. <br> Let's make a conservative assumption that we can randomly select from our subset, include these into the grid, and then do a test to see if we have solved the puzzle in 1 nano-second (using 1 core), and just because we can, let's say we have a 12-core Mac Pro, so we can do 12 selections at once if we fork our code properly.
#
# So, brute force style it will take $10^{50}$ seconds to go through all the possible combinations. Let's also say, we get lucky and we only go through 1% of the combinations before we find the result. So, exactly how long is $10^{48}$ seconds? Well, the age of the Universe is roughly $4 \times 10^{17}$ seconds...see, a terrible idea!

# We've already said that random numbers are the solution here, we just need to be smarter in the way that we do our trials. We know that we achieve this in SA, by comparing the system energy of our previous guess, and comparing this to the energy of our new proposal. So what is the 'energy' in terms of a sudoku puzzle?<br>
#
# Clearly, there is an absolute solution here, so we would expect that $E=0$ for the solution. And $E>0$ for any other incorrect 'solution'. But what makes one solution more wrong than another (i.e., $E>>0$)? <br>
#
# Let's consider what a correct solution would look like. It would have only unique integers in each row, column and 3x3 box. So, if in a particular row, we had the same number twice, then this must be wrong, and this row would not have $E=0$, but $E=1$, i.e., there is 1 incorrect number. So, the energy of our system is simply the sum of all non-unique numbers in each column, row and 3x3 box. Conversely, we can think of this as the number of unique elements in each row, column, and box, and then we get back to $E=0$ by susbtracting what the enrgy of a correct solution would be, $9 \times 9 \times 3 = 243$, <br>
# <center>$ E_{\rm tot} = 243 - \sum_{i,j,k} E_{\rm row,i} + E_{\rm col,j} + E_{box,k}$</center>

# In[141]:


def check(i,k,n):
    #determines number of unique elements in each row (k=1) or column (k!=1)
    nums=np.arange(9)+1
    nu=0
    for j in range(9):
        if k==1:
            i1 = n[i][j]-1
        else:
            i1 = n[j][i]-1

        if i1==-1: # Quick check that there are no zeros
            return -1

        if nums[i1]!=0:
            nu+=1
            nums[i1]=0

    return nu

def checksq(isq, jsq, n):
    # determines number of unique elements in square isq, jsq
    nums=np.arange(9)+1
    nu = 0
    i0 = isq*3
    j0 = jsq*3
    for i in range(3):
        for j in range(3):
            i1 = n[i0+i][j0+j] - 1

            if i1==-1: # Quick check that there are no zeros
                return -1

            if nums[i1]!=0:
                nu+=1
                nums[i1]=0

    return nu

def calc_energy(n):
    # computes the "energy" of the input grid configuration
    sum = 0
    for i in range(9):
        sum += check(i,1,n) + check(i,2,n)

    for i in range(3):
        for j in range(3):
            sum += checksq(i,j,n)

    return 243 - sum


# It's important to remember which are the fixed numbers specific to this puzzle, these cannot be moved between interations. We can just use a mask for this.

# In[142]:


n = sudoku_hard
mask = n>0


# Now, we need a starting position. For this we can take the missing integers from our sudoku example, and apply them pseudo-randomly to the grid. Why pseudo-randomly? We know enough about the final configuration that we can be clever in the way that we distribute the remaining values. We know that the values {1,2,3,4,5,6,7,8,9} must appear in each 3x3 box, so we should only populate those 3x3 squares with those numbers minus those which already exist from this set.

# In[143]:


# Assign the remaining integers to open cells so that each block contains all integers once only.
for ib in range(3):
    for jb in range(3):

        nums=np.arange(9)+1  #{1,2...8,9}
        for i in range(3):
            for j in range(3):
                i1 = ib*3 + i
                j1 = jb*3 + j

                if n[i1][j1]!=0:
                    ix = n[i1][j1]
                    # replace the already used number with 0
                    nums[ix-1]=0

        # Now we have an array that contains the unused numbers.
        # So we populate using that array.
        iy = -1
        num1=np.zeros(sum(nums>0))
        for k in range(9):
            if nums[k]!=0:
                iy+=1
                num1[iy] = nums[k]

        kk = 0
        for i in range(3):
            for j in range(3):
                i1 = ib*3 + i
                j1 = jb*3 + j
                if(n[i1][j1]==0):
                    n[i1][j1] = num1[kk]
                    kk+=1

plot_sudoku(n)


# In[144]:


# What is the energy of our starting position?
e_prev = calc_energy(n)
print 'The system Energy of the initial configuration is:', e_prev

temp = 2.0 # starting temperature
ntrial = 100000 # number of trials we are going to run
thermo= 0.9 # thermostat for the temperature when it's too hot
reanneal=1000 # how often do we apply the thermostat?


# How should we decide how to construct a proposal? If we are going to smart about this and use SA, then the proposal shouldn't just come out of thin air, it should be based loosely on the accepted grid from the previous iteration. So, we trade numbers. We take a 'movable' number at one grid space and swap it with the number at another grid space. But remember, we distributed the numbers so that in each 3x3 box we had the numbers 1-9. To maintain this, we should make sure that we only swap numbers within the same 3x3 box. We then compute the energy of our proposal grid. If the energy goes down, then we accept. Otherwise toss a coin according to,

# <center>${\rm randU}(1) < {\rm exp}\left[-\Delta E / T\right] $</center>

# In[145]:


for ll in range(ntrial):

    # at each step pick at random a block and two moveable elements in the block
    ib,jb = 3*np.random.randint(3,size=2)

    i1,j1 = np.random.randint(3,size=2)
    while mask[ib+i1][jb+j1]:
        i1,j1 = np.random.randint(3,size=2)

    i2,j2 = np.random.randint(3,size=2)
    while mask[ib+i2][jb+j2] or (i1==i2 and j1==j2):
        i2,j2 = np.random.randint(3,size=2)

    # swap the movable elements and compute the energy of the trial configuration
    propose_n=n.copy()
    propose_n[ib+i1][jb+j1] = n[ib+i2][jb+j2]
    propose_n[ib+i2][jb+j2] = n[ib+i1][jb+j1]

    # calculate the proposal system energy
    e_new = calc_energy(propose_n)
    deltaE = e_prev - e_new

    # Check the SA criterion
    if e_new < e_prev:
        e_prev = e_new
        n = propose_n
    else:
        if np.random.rand() < np.exp( float(deltaE)/temp):
            e_prev = e_new
            n = propose_n

    # stop computing if the solution is found
    if e_prev==0:
        break

    if(ll % reanneal) == 0:
        temp=temp*thermo
        if temp<0.1:
            temp=0.5

    # is the code still running properly...?
    if(ll % 5000) == 0:
        print ll,np.exp( float(deltaE)/temp),e_prev,e_new

if e_prev==0:
    print "Solution found after", ll, "steps"
    plot_sudoku(n)


# So, how do we improve on this? Can we speed this up? <br>
#
# There are three ways to approach this:<br>
# (1) why limit ourselves to only one swap per iteration, at least at the beginning?<br>
# (2) intial temperature<br>
# (3) thermostat tuning<br>
