#TODO: Add NPC interaction using model

"""
~~~~ Mini Project: Text Adventure ~~~~
Time: ~2/3 Hours (3+ Sessions); sometimes they want to keep expanding their stories.
Questions:
1. What data type does the input function store inside of our player_name variable? i.e. What data type does the function return?
2. To make our greeting variable which is a string, we stuck together 3 smaller strings.
    What is the fancy name for sticking strings together?
3. We also built a more complicated string using different syntax? What was this called?
4. What is the purpose of \n?
5. At decision_one, if the player types capital Y, what happens?
6. How many elifs could we use in a single if statement?
7. Code usually runs from top to bottom, line by line, like a book. We say that code runs BLANK?

BONUS QUESTION: 
Another question goes here?

Augmentation:
It's not simple anymore!

EXTRA BONUS - The Pen Game:
A pot of different colour whiteboard pens became the perfect prop for understanding conditional statements.
#TODO: Description
"""

import random
from time import sleep

# add your name as a variable to present to the player in the welcome message
your_name = "Jake"
print(f"Welcome to {your_name}'s adventure!")

# ask for the player's name and store it in a variable
player_name = input("What is your name?")

# greet the player using the name they provided
greeting = "Hello " + player_name + "! You find yourself at the edge of a forest,\n \
            the dark evergreen trees towering over you."

# ask for the player's first decision in the story
decision_one = input("Do you enter the forest? (y/n)")

# use an if statement to check how the player wants to move forward
# if they step into the forest we continue the story
if decision_one == "y":
    print("You step forward into the forest.\nAnimals chitter as the light behind you fades away.")
    print("After walking through the moss and foliage for a while, the path forks.\nTo the left, it seems to brighten again. To the right, it's impossible to see into the dark.")

    # the second decision
    decision_two = input("Do you go left or right? (l/r)")

    if decision_two == "l":
        print("You approach the bright light, barely able to see.\nYour next step fails to hit the floor as you fall towards the sea.\n**SPLASH**")
        print("GAME OVER!")
        quit()

    elif decision_two == "r":
        print("You step into the darkness. ")
        print("A raspy voice voice can be heard echoing near by.")
        
        decision_three = input("Do you respond or stay silent? (r/s)")
        # optionally include some randomness 
        noise = random.randint(1,10)

        if decision_three == "r":
            print("\"HEELLLOOOOOOO\", you shout.")
            print(".")
            sleep(1)
            print(".")
            sleep(1)
            print(".")
            sleep(1)
            print("A flock of birds fly out and over you, prompting you to flee and run after them.")
            print("GAME OVER")
            quit()

        elif decision_three == "s":

            print(".")
            sleep(1)
            print(".")
            sleep(1)
            print(".")
            sleep(1)
            print("You stay silent.")

            if noise > 5:
                print("You creep forwards, stepping on a branch.\n \"SNAP\"")
                print("Terrified, you sprint back the way you came.")
                print("GAME OVER")
                quit()

            else:
                print("You sneak forwards, the path brightening as you go until a single beam\
                       of sunlight illuminates a large oak tree in front of you.")
                sleep(2)
                print("It has a face?")
                print("The tree describes a path out of the forest, guiding you towards a large golden amulet atop a pedestal.")
                print("You pick it up!\nYOU WIN!")
                quit()

                

        else:
            print("That wasn't an option!")
            print("GAME OVER!")
            quit()

    else:
        print("That wasn't an option!")
        print("GAME OVER!")
        quit()

elif decision_one == "n":
    print("The strange noises coming from inside convince you to turn away.")
    print("GAME OVER!")
    quit()

else: 
    print("That wasn't an option!")
    print("GAME OVER!")
    quit()