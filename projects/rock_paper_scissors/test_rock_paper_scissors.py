from rock_paper_scissors import rock_paper_scissors, OPTIONS
from markov import MarkovChain, flatten

import json
import numpy as np
from datetime import datetime


game_info = {
    "choices":[],
    "scores":[],        
    "winner":[],
    "ground_truth_matrix":None,
    "estimated_matrix":None,

}
n_games = 0

# randomly sample transition probs from a normal distribution
auto_transition_matrix= np.square(np.random.randn(len(OPTIONS), len(OPTIONS)))
auto_transition_matrix= auto_transition_matrix/auto_transition_matrix.sum(axis=1)[:,None]

# uniform dist examples
auto_transition_matrix = np.array([[1/3]*3 for _ in range(3)])

while True:
    print(f"##### GAME {n_games+1} #####")
    if not game_info["choices"]:
        # run an instance of the game using random choices
        game_info["choices"].append([OPTIONS.index(choice) for choice in rock_paper_scissors(automatic=True, auto_transition_matrix=auto_transition_matrix)[0]])
    else:
        # use the sequence of choices from the previous game(s) to model the player's decisions as a Markov Chain
        chain = MarkovChain(states=[i for i in range(len(OPTIONS))])
        # very naively flatten all choices across games into a single list. Adds in false transitions 
        # as it may be wrong to model the first decision of a new game as dependent on the last decision of the previous game.
        # also causes the problem of tending towards a uniform distribution as n_games increases
        chain.calculate_transition_matrix(flatten(game_info["choices"]))
        player_choices, scores, winner = rock_paper_scissors(automatic=True, auto_transition_matrix=auto_transition_matrix, decision_model=chain)
        game_info["choices"].append([OPTIONS.index(choice) for choice in player_choices])
        game_info["scores"].append(scores)
        game_info["winner"].append(winner)
        
        n_games +=1
        if n_games >= 25:
            break

print("Final Learned Transition Matrix:")
print(chain.transition_matrix)
game_info["estimated_matrix"] = chain.transition_matrix.tolist()
print("Player Transition Matrix:")
print(auto_transition_matrix)
game_info["ground_truth_matrix"] = auto_transition_matrix.tolist()
time = datetime.now().strftime("%d%m%y_%H%M%S")
path = f"projects/rock_paper_scissors/game_history/auto_game_history{time}.json"
with open(path, "w") as f:
    json.dump(game_info, f)