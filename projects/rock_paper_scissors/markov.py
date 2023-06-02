from itertools import product
import numpy as np

class MarkovChain:

    def __init__(self, states, transition_matrix=None):

        self.states = states
        self.current_state = None
        self.transition_matrix = None

        if transition_matrix is not None:
            self.transition_matrix = np.array(transition_matrix)

    def calculate_transition_matrix(self, sequence:list):
        combs = product(self.states,repeat=2)

        transition_counts = {c:0 for c in combs}
        total_counts = {s:0 for s in self.states}

        # count occurrences of state transitions
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            transition = (current_state, next_state)
            
            transition_counts[transition] += 1
            total_counts[current_state] += 1

        # calculate probabilities
        dim = len(total_counts)
        transition_matrix = np.zeros((dim, dim))
        for i, current_state in enumerate(total_counts):
            total_count = total_counts[current_state]
            for j, next_state in enumerate(total_counts):
                transition = (current_state, next_state)
                if transition in transition_counts:
                    transition_count = transition_counts[transition]
                    if total_count:
                        transition_matrix[i, j] = transition_count / total_count
                    else:
                        # if probability for a known state is 0, make all states equally probable
                        transition_matrix[i,j] = 1/dim


        self.transition_matrix = transition_matrix
        return transition_matrix, list(total_counts.keys())

    def set_initial_state(self, initial_state):
        self.current_state = initial_state

    def get_next_state(self):
        if self.current_state is None:
            raise ValueError("Initial state not set.")

        if self.transition_matrix is None:
            raise ValueError("No transition matrix. Either pass one to init\
                              or calculate one from a sequence using MarkovChain.calculate_transition_matrix()")
        
        transition_probabilities = self.transition_matrix[self.current_state]
        next_state = np.random.choice(self.states, p=transition_probabilities)
        self.current_state = next_state
        return next_state
    
def flatten(l):
    return [item for sublist in l for item in sublist]