# **:mage: MYSTIC :mage:**
*(**M**aterials for **Y**oung **S**tudents and **T**eachers to **I**mprove and **C**ode)*

----
**WIP 12/06/2023**
Will be updated regularly as I tidy and refactor the additional material

----
MYSTIC is a collection of materials that effectively combines **self-teaching** and **teaching others**. The materials were originally used to teach Python fundamentals to a class of primary school children, whilst simultaneously providing a weekly challenge to the person delivering the course (i.e. me).

If you use or find this helpful please let me know!

## **:brain: The Concept :brain:**
```
Create accesible weekly Python material.

Augment the material with a concept that you as the course deliverer would like to improve/consolidate your understanding of.
```

### For example:

**Material** - Simple Rock, Paper, Scissors Game

**Augmentation** - Model the User's Inputs as a [Markov Chain](https://www.youtube.com/watch?v=i3AkTO9HLXo)

How is this useful?
- Introduces contextualised programming concepts to the pupils.
- The base material is simple, but is framed as important with respect to the full system.
- The augmented material imports the original material, demonstrating code structure, modules and packages.
- The added complexity encourages students to want to learn *how* and *why*
- If you can explain what's happening to a 10 year old, you understand what you're talking about (to a degree)

----
## The Material ##

**1. [Rock, Paper, Scissors](projects/rock_paper_scissors/)**
    
    A simple simulation of rock, paper, scissors. The user plays against the computer in a Best of N. The computer opponent has 3 modes: random, impossible and smart. The smart mode incorporates a simplified Markov Model to attempt to naively learn from the player's previous decisions.

- Materials: `rock_paper_scissors.py`, `markov.py`

**2. [Text Adventure](projects/text_adventure/)**

    A classic choose-your-own-adventure style setup that focuses on conditional statements and nesting them to get the desired paths within your code. Most decision points are simple 2 option forks with yes/no, left/right questions etc. The improved version also allows for natural language interaction with "characters" using a 'Stupid Transformer' model to predict player sentiment and fork paths based on the output. The entire system uses only numpy.
    
- Materials: `text_adventure.py`, `pos_neural.py`, `inference.py`

**3. [Treasure Hunt](projects/treasure_hunt/)**

    An introduction to nested data structures like lists-of-lists. A treasure map is established as an NxN grid filled with zeros. Certain coordinates are randomly replaced with treasure. Pupils' characters can be set on a random walk around the grid collecting treasure and can be left running in the background of the lesson. The teacher's character has a 'map' and approximates a Travelling Salesman algorithm to attempt to collect all the treasure before the hoarde of pupils. 

- Materials: `treasure_hunt.py`, `travelling_salesman.py`

