from bpe import BPE
from pos_neural import SeqNeuralNet
import random
import numpy as np 

class NPC:
    """
    An NPC whose mood depends on natural language of the player.

    How it works:
        - The player's sentence is parsed by a pretrained BPE tokenizer returning token ids
        - The token ids are passed to a neural architecture
            - The first layer of the net is an Embedding Layer
                this is an in-line embedding layer that is learned jointly during training of the full net.
            - The following layers are dense feedforward layers with an elu activation
            - The final layer is a single regression output value
        - The sentiment output is used to look up a list of responses from which a response is randomly chosen and printed.
        - The sentiment output is passed to the conditional in the story to choose the plot path.
    """

    def __init__(self) -> None:

        self.sentiment_num = None
        self.sentiment_str = None

        self.dialogue_prompts = [
                                    "Greetings, adventurer! What do you think of my magnificent collection of enchanted hats?",
                                    "Ah, hello there! How do you feel about the craftsmanship of my legendary golden sword?",
                                    "Welcome, brave soul! What's your opinion on the rare and mystical gemstone I possess?",
                                    "Hello, traveler! Care to share your thoughts on the elegance of my prized antique pocket watch?",
                                    "Well met, young hero! How do you find the aroma of my collection of magical potions?",
                                    "Salutations, wanderer! What's your take on the intricately woven tapestries adorning my walls?",
                                    "Hail, my friend! Tell me, what do you think of the mesmerizing melodies played by my enchanted flute?",
                                    "Greetings, noble adventurer! How does my shield, adorned with ancient symbols, strike you?",
                                    "Ah, good day! What's your opinion on the shimmering gem-encrusted amulet around my neck?",
                                    "Welcome, brave traveler! How do you feel about the comfort and style of my custom-made boots?",
                                    "Well hello, good sir! What do you make of this mystical creature I keep as a pet?\n*SHOWS YOU A SMALL STRANGE BRIGHT YELLOW MOUSE THAT CRACKLES WITH ELECTRICITY*",
                                    "Morning, traveller! Welcome to my perfumery! How does it smell?\n*THE STRONG AROMA OF LAVENDER ATTACKS YOUR NOSE*",
                                ]


        self.responses = {
                            "positive": [
                                "Wow, that's really made my day!",
                                "I'm really glad you said that!",
                                "Thank you so much, I needed to hear that!",
                                "That's incredibly kind of you!",
                                "You've really brightened my mood!",
                                "I appreciate your uplifting words!"
                            ],
                            "neutral": [
                                "Right, okay...",
                                "I see...",
                                "I understand what you're saying.",
                                "I suppose you have a point.",
                                "I'm processing what you just said.",
                                "Interesting perspective."
                            ],
                            "negative": [
                                "Why would you say that to me? :(",
                                "That's really hurtful...",
                                "I'm sorry if I offended you.",
                                "Ouch, that's harsh.",
                                "Your words have really hurt me.",
                                "I didn't expect that from you."
                            ]
                        }

        
        self.bpe = BPE.from_pretrained()
        self.model = SeqNeuralNet.from_pretrained(path="projects/text_adventure/models/seqnet_linear_enhanceddata.pkl")

    def greet(self):
        return random.choice(self.dialogue_prompts)

    def inference(self, user_input):
        # tokenize the input. Model was trained on max_len=100
        tokenized_input = self.bpe.tokenize_to_ids(user_input, max_len=20)
        # pass it to the model
        return self.model.predict(tokenized_input)


    def respond(self, user_input, thresholds={"positive":0.6, "neutral":0.3, "negative":0}):
        self.sentiment_num = self.inference(user_input)
        
        if self.sentiment_num > thresholds["positive"]:
            self.sentiment_str = "positive"

        elif self.sentiment_num > thresholds["neutral"]:
            self.sentiment_str = "neutral"

        else:
            self.sentiment_str = "negative"

        response = random.choice(self.responses[self.sentiment_str])

        return self.sentiment_str, response
