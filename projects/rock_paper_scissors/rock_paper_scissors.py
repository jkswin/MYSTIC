"""
~~~~ Mini Project: Rock Paper Scissors ~~~~
Time: ~1hr30 (2 Sessions)
Questions:
1. What is the data type of our OPTIONS variable?
2. What arguments does the rock_paper_scissors() function take?
3. What kind of loop do we use in the rock_paper_scissors() function? Could we use a different type of loop?
4. What does the continue keyword do?
5. What module have we imported? How is it used?
6. Why do we define user_wins and bot_wins outside of our loop?
7. What is the difference between = and ==?
8. Is this code the only way of making a rock, paper, scissors game in Python?

BONUS QUESTION: 
Recall that when we use the import keyword, we are accessing code from a different python file, written by us or someone else. 
In this example, we imported random and called random.choice(). The code in the other file that we borrowed from did not run, it simply gave us access to a function.
When we import in this way, we want to make sure that the contents of that file only runs when we want it to. e.g. when we use a particular function.

Using this information:
What do you think is the purpose of `if __name__ == "__main__"`?

Augmentation:
Regular rock,paper,scissors is heavily studied from a psychological perspective. Playing against a random machine is different; players likely (or should) aim to be random, 
but humans are terrible at being random, and often fall into predictable patterns when trying to do so. I naively model the players' decisions as Markov Chains and assume that the only
factor affecting their decision is their previous choice. This information is passed to the bot to help choose which option will most likley beat the players.
"""


import random

# list all of the options available when playing rock paper scissors
OPTIONS = ["rock", "paper", "scissors"]

# using a dictionary we can make a mapping of 'key beats value'
beats = {"rock": "scissors",
         "paper": "rock",
         "scissors":"paper"}

# and 'key loses to value'. In our function this lets us check who wins like this: loses_to["rock"] -> "paper"
loses_to = {"rock":"paper",
            "paper": "scissors",
            "scissors": "rock"}

# define our function for running the game
def rock_paper_scissors(best_of=10, unwinnable=False, decision_model=False):

    # establish the number of games needed to win based on 'best of'
    wins_needed = best_of//2 + 1
    # define variables to keep track of how many wins the user and computer have
    user_wins = 0
    bot_wins = 0

    player_choices = []

    # little welcome message
    print("~~~~ Welcome to Rock Paper Scissors ~~~~")
    print(f"This is a best of {best_of}. The first to {wins_needed} wins!")
    print("Write rock, paper or scissors:")
    
    # loop through games of rock, paper, scissors until a winner is declared
    while True:

        # collect the decision of the user. i.e. what position their imaginary hand is in
        user_hand = input().lower().strip()

        # make sure it's a valid choice
        if user_hand not in OPTIONS:
            print(f"Please type one of: {OPTIONS}")
            continue
        
        
        # (add an option to make the computer always choose the winning response) '-'
        if unwinnable:
            bot_hand = loses_to[user_hand]
        else:
            # otherwise randomly choose one of rock, paper or scissors
            bot_hand = random.choice(OPTIONS)

        if decision_model and player_choices:
            decision_model.set_initial_state(OPTIONS.index(player_choices[-1]))
            bot_hand_idx = decision_model.get_next_state()
            player_pred = OPTIONS[bot_hand_idx]
            bot_hand = loses_to[player_pred]

        # print what each participant chose    
        print(f"{user_hand.upper()} vs {bot_hand.upper()}")
        player_choices.append(user_hand)

        # if both choose the same e.g. scissors and scissors, it's a draw
        if user_hand == bot_hand:
            print("DRAW")
        
        # if the user's hand beats the bot's hand, add 1 to the user's win count
        elif beats[user_hand] == bot_hand:
            print("WIN")
            user_wins +=1

        # otherwise check if it loses. This could also just be an else in this context
        elif loses_to[user_hand] == bot_hand:
            print("LOSE")
            bot_wins +=1 

        print(f"USER: {user_wins}/{wins_needed} BOT: {bot_wins}/{wins_needed}")

        # after each game, check if either of the players have passed the wins threshold
        if bot_wins >= wins_needed:
            print("The bot wins!\nGAME OVER")
            break

        elif user_wins >= wins_needed:
            print("You win!")
            break

        print("\n\n")

    # return for the sake of the decision model, don't add to taught implmentation
    scores = [user_wins, bot_wins]
    winner = int(np.argmax(scores))
    return player_choices, scores, winner


### AUGMENTATION ###

import numpy as np
from itertools import product
import json

class MarkovChain:
    def __init__(self, states, transition_matrix=None):

        self.states = states
        self.current_state = None
        self.transition_matrix = None

        if transition_matrix:
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


if __name__ == "__main__":

    game_info = {
        "choices":[],
        "scores":[],        
        "winner":[],

    }
    while True:
        if not game_info["choices"]:
            # run an instance of the game using random choices
            game_info["choices"].append([OPTIONS.index(choice) for choice in rock_paper_scissors()[0]])
        else:
            # use the sequence of choices from the previous game(s) to model the player's decisions as a Markov Chain
            chain = MarkovChain(states=[i for i in range(len(OPTIONS))])
            # very naively flatten all choices across games into a single list. Adds in false transitions 
            # as it may be wrong to model the first decision of a new game as dependent on the last decision of the previous game.
            # also causes the problem of tending towards a uniform distribution as n_games increases
            chain.calculate_transition_matrix(flatten(game_info["choices"]))
            player_choices, scores, winner = rock_paper_scissors(decision_model=chain)
            game_info["choices"].append([OPTIONS.index(choice) for choice in player_choices])
            game_info["scores"].append(scores)
            game_info["winner"].append(winner)
            
            ui = input("Continue?\nn -> Quit\nm -> See Transition Matrix\nh -> See Game History")
            if ui == "n":
                break
            elif ui == "m":
                print(chain.transition_matrix)

            elif ui == "h":
                print(game_info)
    
    path = "game_history.json"
    with open(path, "w") as f:
        json.dump(game_info, f)
