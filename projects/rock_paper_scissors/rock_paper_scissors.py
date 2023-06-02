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
import numpy as np
import json
from markov import MarkovChain, flatten

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

# define our function for running the game. When teaching, only include the best_of and unwinnable args
def rock_paper_scissors(best_of=10, unwinnable=False, decision_model=False, automatic=False, auto_transition_matrix=None):

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

    if automatic and auto_transition_matrix is not None:
        auto_chain = MarkovChain(states=[0,1,2], transition_matrix=auto_transition_matrix)
    
    # loop through games of rock, paper, scissors until a winner is declared
    while True:

        # collect the decision of the user. i.e. what position their imaginary hand is in
        if automatic and auto_transition_matrix is not None:
            #### AUG ####
            # if we are running the game so that it is bot vs bot
            if player_choices:
                auto_chain.set_initial_state(OPTIONS.index(player_choices[-1]))
                user_hand = OPTIONS[auto_chain.get_next_state()]
            else:
                user_hand = random.choice(OPTIONS)
            ############
        else:
            # if a real person is playing, take their input
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

        ##### AUG ######
        if decision_model and player_choices:
            decision_model.set_initial_state(OPTIONS.index(player_choices[-1]))
            bot_hand_idx = decision_model.get_next_state()
            player_pred = OPTIONS[bot_hand_idx]
            bot_hand = loses_to[player_pred]
        ################

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


# run example 
if __name__ == "__main__":
    rock_paper_scissors(best_of=5,
                        )
    
    rock_paper_scissors(best_of=5,
                        unwinnable=True,
                        )
    