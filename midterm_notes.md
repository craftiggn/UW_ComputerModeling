# 1
- Book "The general and logical theory of automata" -- 1948.
- Automata is defined on a grid of cells, each of which can attain a discrete number of states.
- EVOLUTION -- cellular automata evolves according to the transition rules.
- Neighbourhoods, in 2d:
    - von Neumann i.e. 4 cells
    - Moore i.e. 8 cells
- Game of life:
    - cells remains alive if it has 2 or 3 live neighboors (Moore)
    - cell becomes alive if it has exactly 3 live neighboors
    - other cases the cell dies / remains dead
- Limit Cycle -- existence of limit cycles --> IRREVERSIBILITY.
- rulesets can be reversible or irreversible.
- Construction of reversible automata: add an input for the cell's state in (t-1) and mod 2.
- reversible rulesets have 'R' at the end.
<!-- Ask chat about this -->
- Macroscopic analoguy: many-body systems -- irreversible macroscopic behaviour governed by reversible classical mechanics.
- Classification of 1D automata:
    - Class 1 -- end up in a uniform state
    - Class 2 -- end up in a fixed point or a periodic orbit
    - Class 3 -- chaotic behaviour (sensitivity to initial conditions)
    - Class 4 -- "On the edge of chaos" in which the organized structures appera which propagate and interact with each other
- Automata can be used to create: simple calculators, logical operations, TURING MACHINE.

# 2
- Turing Machine -- A computer aparatus, consisting of a head (which can be in one of M discrete states) and a tape divided into cells which can be in one of N discrete states. Depending on the combination of the head state and the cells state the machine writes the new value on the tape, changes its state, and can also move the tape.
- Halting Problem -- Determine whether a Turing machine will, with a given code, stop after finite iterations or go on forevere. It is, in general undecidable, there is no general algoithm.

Formalism: 
    - We are given as an input the code of a program P and a test input x.
    - We want to know whether P, on input x, halts or not.

- Universal Turing machine -- a TM, which is capable of emulation the workings of any other Turing .machine (computes everything) - by means of a long enough code.
- Automaton 110 (2D) is a universal turing machine.
    - Gliders -- used for computation, there are 3 types.

# 3 Particle-based tissue modeling
- Soft Matter Physics -- mesoscale --> tissues and organoids, up to 1mm.
- Dense Active Matter -- active fluids.
- Active glass -- an amorphous material which can go into a reversible transition from a "solid-like" into a more viscus "liquid-like" state, when "activity" is increased.
- Active drivers in cell monolayers: cell migration, division, ingression, chemical signaling, cell-type interactions, junction contractions, shape fluctuations, boundaries and topology.
- Cell division as a main driver for morphogenesis, active drivers reduced to: cell migration, division, ingression (removal)
<!-- Some compnonets of this lecture are messy, not noted, look in lab. -->

# 4 Thin sheets: membranes and wrinkling
- Membranes -- assemblies or aggregates of molecules (atoms) which have formed very thin sheets.
- Wrinkles are a common features of membranes.
- Why do wrinkles form?
    - they can help in creating oxygen gradient within the biofilm
    - can provide an enhanced system for nutrient and waste transport
- Physics of thin membrances:
    - natural surface -- neither stretches or compresses (in the middle of the sheet)
    - Elastic Energy -- depends on the strain tensor, can be divided into stretching and bending
    - stress pararell to the neutral surface can be neglected -- state of plane-stress
    <!-- UNCLEAR -->
    - points which lie on a normal to the neutral surface in the reference configuration remain on the same normal in the deformed configuration
<!-- maybe write down the equations for this lab? -->

# 5 Shortest path optimizations
- traveling salesman problem
- Solutions:
    - Brute force:
        - Breadth first -- keep adding nodes that are the closest to the starting node (BAD)
    - Swarm intelligence -- ants
        - use pheromone trails
        - the shortest path contains more pheromones due to high traffic
        - pheromones evaporate, but are reinforced with more passing ants
        - edges have weights (length - based)
        <!-- confirm this -->
        <!-- write down the equations -->
        <!-- what is the weight here? -->

# 6 Things in motion
- sheep or other large bodies of individuals exhibit behaviour similar to fluid when moving
- Reynolds model:
    1. Separation -- Move to avoid crowding local flockmates
    2. Alignment -- Match orientation and velocity of the neighbours
    3. Cohesion -- Move twoards the average position of the neighbours
- Self organisation of the flock --> Emergence (large complex structures appear).
- Applications: Animations, Drones.
- Can be used to model traffic --> Nagel Schreckenberg model:
    - accelerate -- if you can do it
    - breaking -- if you are in danger of collision, reduce speed to a safe one
    - randomness -- from time to time randomly slowdown (absent-minded driver)
- Phantom traffic jams --> Nagoya experiment (driving in a circle)
- spontaneous traffic jams propagate in the direction opposite of the car motion witha velocity of 15 kph (universal)
<!-- draw the fundamental diafram! -->
- Pedestrian model:
    - external factors (fire etc.)
    - interaction with othe obstacles (repulsion)
    - attraction towards the target
- PM --> Lane formation -- self-organisation

# 7 Evolution and genetic algorithms
- Evolution in 4 steps:
    1. initial population
    2. natural selection
    3. reproduction
    4. mutation
- Evolutions is a consequence of: heredity (memory), variablity (mutations), more individuals than can survive (surplus).
- Uniquness -- evolution acts on a very small subset of all possible individuals, new variants arise by small variations of what was before. If life started again, the outcome would be entirely different.
- Evolution is a very successfull model finding nearly optimal solutions to complex problems
- Genetic algorithms mimic evolution in a way that:
    - it finds a nearly optimal solution in a high-dimensional landscape
    - is usefull when the problem is too complex, and searching through a solution space is difficult
    - the aim isn't perfection, rather a good, feasable solutions in an acceptable timescale
    - there is no guarantee that a perfect solution can even be found (and we might not even know if the algorithm will work)
- Genetic algorithm features:
    - implemented in computer simulation
    - individuals represented by abstract structures called chromosomes
    - evolution usually starts from a population of ranomly generated individuals
    - in each generation the fitness of every individual is evaluated -- according to which some are killed and some reproduce
    - individuals can reproduce sexually, by cloning, or any other means. Frequency of mutations is a crucial factor (usually very low)
    - Max fitness of a population over time expressed as a graph tends to have these elements: slow evolution, rapid jump, stasis.
- Punctuated equilibrium -- species are generally morphorogically stable, changing little for millions of years. This leisurely pace is "punctuated" by rapid burst of change that results in a new species.
- Dynamic landscape -- due to cooperativity the landscape itself evolves with the species, concept of fitness becomes questionable. This can increase stability but can also lead to great extinctions when this stability is finally disturbed.

# 8 Neural networks
- Human learning should be represented as strengthening of some property of neurons, XX century --> 1949 Donald Hebb
- "Neurons that fire together, wire together. Neurons that fire out of sync, fail to link." -- Hebb's rule
- Randomly destroying small percentages of the population does not affect the newtork's performance
- Brains are very efficient
- Spin glass model - Hopfield network - Model hugely inspired by Ising model of ferromagnetism
<!-- write a note on this model - equation -->

# 10 Computer Modeling with single prompts
- Marker and cell method -- Full, explicit Navier-Stokes solver
- Bitgram -- simple language model -- Build probability table from large text (prob. of next letter based on current letter)
- LLM
    - Tokenizer
    - Neural network
    - large context
- prompting - art of getting the llm do what you want it to do
- "deep thinking" -- llm coverses with itself

# 10 Lattice Gasses & Lattice Boltzmann Method
- Mesoscopic approach -- between Macro and Micro, Ensamble everything
- Lattice-particle methods -- Solve fluid dynamics using fictitious particle dynamics, Molecular details do not cound as long as correct dynamics is recovered in the macroscopic limit
- Be wise, discretize! -- Marek Kac
- Lattice gas automata:
    - streaming and collisions
    - positions restricted to lattice sites
    - discrete velocities
    - no two particles with the same velocity allowed at one site
    - Collisions -- defined output sate on given input state
    - Pros: simple, fast, exact (no rounding), stable
    - Cons: Noisy, viscosity cannot be tuned
- Von Karman street 
- Lattice Boltzmann model:
    - initial state -> post colision -> propagation 
    - discreete set of velocities
    - Hydrodynamic fields are moments of the distribution function
    - evolution equation: f_i(t+1) = f_i(t) - [f_i(t) - f_eq(t)] / Tau
    - f_eq = equilibrium distribution -- a small u expansion of he local maxwell distribution: start from the maxwell dist. -> normalize the velocities -> expand in u up to O(u^2)
    - the weights W_i are determined from the isotropy conditions
    - external forcess are incorporated in a velocity term
    - Solid-fluid boundaries -> Bounce-back rules 

- Summary:
Lattice-Boltzmann method is
• a computational approach based upon particle movement
and collisions.
• easy to implement and parallelize thanks to the local
dynamics
• simple, flexible and able to incorporate complex boundary
conditions
It may also be used to simulate the fluid with immersed
moving objects (suspensions, polymers etc.)

# 11 Game Theory
- The **rules** of the game state who can do what, and
when they can do it
– A player's **strategy** is a plan for actions in each
possible situation in the game
– A player's **payoff** is the amount that the player wins
or loses in a particular situation in a game
– A players has a **dominant strategy** if his best
strategy doesn’t depend on what other players do

- Zero-sum - situation in which a participant's gain or loss is exactly balanced by the losses or gains of the other participants
- Nash Equilibrium - a situation when each player's strategy is optimal, given the strategies of the other players
- Every finite game has at least one Nash equilibrium in either pure (deterministic) or mixed (probabilistic) strategies
- payoff matrix - in lecture slides

# 12 Social Networks
- graph theory - Euler
- degree - the number of edges connected to a node
- homophily - people tend to connect with people with similar beliefs
- heterophobia - people tend to isolate themselfs from people with differing beliefs
- social contagion - people tend to assume beliefs of their company as their own after elongated exposure

# 13 River Networks