# imports
import math
import itertools
import time
import matplotlib.pyplot as plt
import random
random.seed(1) # Setting random number generator seed for repeatability

NUM_NEURONS = 10000
NERVE_SIZE = 128000 # nanometers
CONFLICT_RADIUS = 500 # nanometers

##############################################################################
######################## Functions for both algorithms #######################
##############################################################################

def check_conflict(neuron1, neuron2, conflict_radius):
    ''' Computes the distance between two neurons and checks if they are in 
    conflict as defined by conflict_radius.
    '''
    
    distance =  math.sqrt((neuron1[0]-neuron2[0])**2 + (neuron1[1]-neuron2[1])**2) 
    
    if distance <= conflict_radius:
        in_conflict = True
    else:
        in_conflict = False
    
    return in_conflict

def get_neuron_key(neuron):
    ''' Get the key xy of a neuron from its location given by [x, y].
    '''
    return str(neuron[0]) + str(neuron[1])

##############################################################################
############################## Algorithm 1 ###################################
##############################################################################

def check_for_conflicts_1(nerves, conflict_radius):
    ''' Algorithm 1 to the solve the problem.
     
    The distance between all unique combinations of neurons are computed and 
    checked for conflict. 
    
    Complexity: number of combinations dependent on C(n,k) = n!/(n-k)!k!, where 
    n = number of samples and k = number of objects in combination (k=2 in this case),
    therefore algorithm is O(C(n,k)). For k=2, this is faster than an O(n^2) algorithm. 
    '''
    
    start_time = time.time()
    
    neurons_in_conflict = {} # dictionary of conflicted neurons
        
    combos = itertools.combinations(nerves, 2) # generate list of all unique combos
    
    for combo in combos: 
        neuron1 = combo[0] 
        neuron2 = combo[1]
        # check for a conflict
        in_conflict = check_conflict(neuron1, neuron2, conflict_radius)
        if in_conflict:
            # get key for each neuron
            neuron1_key = get_neuron_key(neuron1)
            neuron2_key = get_neuron_key(neuron2) 
            # update conflict dictionary
            neurons_in_conflict[neuron1_key] = True
            neurons_in_conflict[neuron2_key] = True
                
    end_time = time.time()
    total_time = end_time - start_time
    
    return len(neurons_in_conflict), total_time

##############################################################################
############################## Algorithm 2 ###################################
##############################################################################

def get_square_key(neuron, conflict_radius, x, y):
    ''' Gets the position key of square in which a given neuron resides.
    '''
    x_neighbour = neuron[0] + (x * conflict_radius) # x coordinate of neuron in same position in square, but in neighbour square
    x_neighbour_square = x_neighbour / conflict_radius # for coordinate x_neighbour, convert to square position  
    x_neighbour_square = math.floor(x_neighbour_square) # gets position of top left corner of square
    x_neighbour_square = str(x_neighbour_square)
    
    y_neighbour = neuron[1] + (y * conflict_radius) 
    y_neighbour_square = y_neighbour / conflict_radius 
    y_neighbour_square = math.floor(y_neighbour_square) 
    y_neighbour_square = str(y_neighbour_square)
    
    # get square position
    square_pos = 'x_'+x_neighbour_square+'_'+'y_'+y_neighbour_square
    
    return square_pos

def get_squares(neuron, conflict_radius):
    ''' Returns a list of the keys of squares surrounding the neuron and the square the
    neuron is in.
    '''
    squares = []
    for i in range(-1, 2): # -1, 0, 1
        for j in range(-1, 2): # -1, 0, 1
            # get the location of the specified square e.g. x=-1, y=-1 is the square left 1 and up 1
            square = get_square_key(neuron, conflict_radius, i, j)
            squares.append(square)
    return squares

def check_for_conflicts_2(nerves, conflict_radius):
    ''' Algorithm 2 to solve the problem using hashing and rounding technique.'
    
    Split the NERVE_SIZE x NERVE_SIZE nerve bundle into squares of size CONFLICT_RADIUS.
     - Each square has it own unique key, e.g square x_20_y_50 is the 20th square across, and 50th down
     - For each neuron, obtain key for the square it resides in and the keys for 
       the surrounding squares
     - For each neighbouring square, check for other neurons
     - Check for conflict between neuron and any neighbouring neurons
     
    Complexity: complexity is greaty reduced in this algorithm as for each neuron,
    conflict is only checked with neurones in the same and neighbouring squares. 
    As n increases, the number of checks for conflicts increases proportionally, 
    thus the algorithm can be considered O(n).
    '''
    
    start_time = time.time()
        
    populated_squares = {} # dictionary of squares with a neuron inside
    neurons_in_conflict = {} # dictionary of conflicts
    
    for neuron in nerves:
        neighbour_squares = get_squares(neuron, conflict_radius)
        for neighbour_square in neighbour_squares:
            if neighbour_square in populated_squares: # if neighbour square has a neuron inside
                neighbour_neurons = [populated_squares[neighbour_square]] # get the neurones in the neighbour square
                for neighbour in neighbour_neurons:
                    in_conflict = check_conflict(neuron, neighbour, conflict_radius)
                    if in_conflict: 
                        # get key for each neuron
                        neuron_key = get_neuron_key(neuron)
                        neighbour_key = get_neuron_key(neighbour) 
                        # update conflict dictionary
                        neurons_in_conflict[neuron_key] = True
                        neurons_in_conflict[neighbour_key] = True
        
        # get the key of the square the current neuron is in
        neuron_square = get_square_key(neuron, conflict_radius, 0, 0)
        
        # update dictionary of populated squares
        if neuron_square not in populated_squares:
            populated_squares[neuron_square] = neuron
        
    end_time = time.time()
    total_time = end_time - start_time
    
    return len(neurons_in_conflict), total_time
        
##############################################################################
########################### Functions to run #################################
##############################################################################


def gen_coord():
    # DO NOT MODIFY THIS FUNCTION
    return int(random.random() * NERVE_SIZE)

if __name__ == '__main__':
    neuron_positions = [[gen_coord(), gen_coord()] for i in range(NUM_NEURONS)]
    # run first algorithm
    n_conflicts_1, time1 = check_for_conflicts_1(neuron_positions, CONFLICT_RADIUS)
    print ("Neurons in conflict: %d, time taken: %.2f seconds" % (n_conflicts_1, time1))
    # run second algorithm
    n_conflicts_2, time2 = check_for_conflicts_2(neuron_positions, CONFLICT_RADIUS)
    print ("Neurons in conflict: %d, time taken: %.2f seconds" % (n_conflicts_2, time2))
        
##############################################################################
###################### Empirical testing of algorithms #######################
##############################################################################

# define list of NUM_NEURONS to run algorithm on
NUM_NEURONS = list(range(1000, 15000, 3000))
alg_1_times = []
alg_2_times = []

# run both algorithms on each NUM_NEURONS and record time 
for num in NUM_NEURONS:
    neuron_positions = [[gen_coord(), gen_coord()] for i in range(num)]
    _, time1 = check_for_conflicts_1(neuron_positions, CONFLICT_RADIUS)
    alg_1_times.append(time1)
    _, time2 = check_for_conflicts_2(neuron_positions, CONFLICT_RADIUS)
    alg_2_times.append(time2)

def plot_computation_times(times):
    ''' Plots the time in seconds to run the algorithm for each given n.
    '''
    plt.plot(NUM_NEURONS, times)
    plt.xlabel('n')
    plt.ylabel('time (s)')
    plt.show()
    
plot_computation_times(alg_1_times) 
plot_computation_times(alg_2_times) 
