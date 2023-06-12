from inference import NPC

while True:
    npc = NPC()
    decision_four = input(npc.greet())
    sentiment, response = npc.respond(decision_four)
    print(response, sentiment, npc.sentiment_num)