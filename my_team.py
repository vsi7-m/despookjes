# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

import math

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # Berekenen wat een tunnel is van tevoren omdat we maar 1 seconde compute time hebben
        # "Is Pacman in a tunnel" 
        self.dead_end_tunnels = {}

        walls = game_state.get_walls()
        width, height = walls.width, walls.height

        def get_free_neighbours(pos):
            x, y = pos
            neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            return [pos for pos in neighbours if not game_state.has_wall(pos[0], pos[1])]

        dead_ends = []
        for x in range(width):
            for y in range(height):
                if not game_state.has_wall(x, y):
                    neighbours = get_free_neighbours((x, y))
                    if len(neighbours) == 1:
                        dead_ends.append((x, y))
        
        for end in dead_ends:
            curr = end
            prev = None
            depth = 1 # lengte van de tunnel
            tunnel_coords = []
        
            while True:
                neighbours = get_free_neighbours(curr)
                if len(neighbours) > 2: 
                    entrance = curr
                    break # om uit de while loop te gaan

                tunnel_coords.append(curr) # coördinaten van de tunnel
                next_coord = [coord for coord in neighbours if coord != prev][0]
                prev = curr
                curr = next_coord
                depth += 1

            for coord, depth in tunnel_coords:
                self.dead_end_tunnels[coord] = entrance



    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action): # Hier wat features aan toevoegen
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action): # Ideale weights zoeken met ML???!!!!
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent): # inspiratie van de slides Approximate Agent (code lijkt daar beetje op)
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        food_list = self.get_food(successor).as_list()
        capsule_list = self.get_capsules(successor)
        opponents_list = [successor.get_agent_state(opponent) for opponent in self.get_opponents(successor)]
        ghost_within_5_list = [
            ghost 
            for ghost in opponents_list
            if (not ghost.is_pacman) and
            ghost.get_position() is not None and 
            ghost.scared_timer == 0
        ] # Misschien beter als we zo typen wanneer tekst niet meer op het scherm past?
        scared_ghosts_list = [
            ghost 
            for ghost in opponents_list
            if (not ghost.is_pacman) and
            ghost.get_position() is not None and 
            ghost.scared_timer > 0
        ]

        agent = successor.get_agent_state(self.index)
        my_pos = agent.get_position()

        # Om food op te eten
        features['remaining_food'] = len(food_list)  # self.get_score(successor)

        width = game_state.data.layout.width
        height = game_state.data.layout.height
        middle = width // 2
        if self.red:
            middle = middle - 1 # middle is het einde van ons eigen territorium, dus waar pacman terug een spookje wordt
        #else: middle = middle + 1 
        # Berekening klopt denk ik wel
        middle_positions = [(middle, h) for h in range(1, height) if not game_state.has_wall(middle, h)] # Als je maze distance berekent met een positie waar een wall is wordt blijkbaar een error gethrowd
        distances_to_home = [self.get_maze_distance(my_pos, pos) for pos in middle_positions] 
        features['distance_to_home'] = min(distances_to_home) # Kortste pad naar positie waar we weer ghost worden

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_food_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_food_distance

        # Compute distance to the nearest capsule (lekker voor als de ghost dichtbij is)
        if len(capsule_list) > 0:
            min_capsule_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsule_list])
            if min_capsule_distance <= 10:
                features['distance_to_capsule'] = min_capsule_distance
        # We moeten dan ook reward geven om een capsule te eten, anders wordt na het eten de afstand tot een andere capsule plots kei groot waardoor pacman de capsule niet wilt eten
        # => Dit blijft een probleem, wrs lukt het wel met de juiste weights maar moeilijk om te vinden
        features['remaining_capsules'] = len(capsule_list)

        # Doodgaan is heel slechttt (dat wist pacman niet als hij zat in de tunnel, hij begon daar gewoon te chillen)
        current_pos = game_state.get_agent_state(self.index).get_position()
        features['death'] = 0
        if my_pos == self.start and current_pos != self.start:
            features['death'] = 1
        
        # Compute distance to the nearest ghost within Manhattan distance 5
        # Als de ghosts scared zijn, hoeven we er niet van weg te gaan (maar mss als de timer net gedaan gaat zijn wel, kunnen we nog verbeteren)
        if len(ghost_within_5_list) > 0 and features['death'] == 0: # Weglopen is alleen belangrijk als we levend kunnen blijven (en dus niet vastzitten in een tunnel)
            min_ghost_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghost_within_5_list])
            features['distance_to_ghost'] = 1 / min_ghost_distance  # 1/ want pacman bij die positieve reward/weight probeerde pacman die 5 distance te houden, 1/ is dus beter met negatieve reward/weight

        # Compute distance to the nearest scared ghost
        if len(scared_ghosts_list) > 0:
            min_scared_ghost_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in scared_ghosts_list])
            closest_scared_ghost = [ghost for ghost in scared_ghosts_list if self.get_maze_distance(my_pos, ghost.get_position()) == min_scared_ghost_distance][0]
            if closest_scared_ghost.scared_timer > min_scared_ghost_distance + 2: # Genoeg tijd geven om de ghost te eten
                features['distance_to_scared_ghost'] = min_scared_ghost_distance
            else:
                features['distance_to_ghost'] = max(features['distance_to_ghost'], 1 / min_scared_ghost_distance) # sws wegrennen van gewone ghost of scared ghost die bijna niet scared gaat zijn
        
        features['ate_scared_ghost'] = 0 # toegevoegd omdat hij scared ghost niet wilde opeten tot het laatste moment
        current_opponents = [game_state.get_agent_state(opponent) for opponent in self.get_opponents(game_state)]
        for ghost in current_opponents:
            if not ghost.is_pacman and ghost.get_position() is not None and ghost.scared_timer > 0:
                if my_pos == ghost.get_position():
                    features['ate_scared_ghost'] = 1

        
            
        # Nu alleen nog maar de distance binnen die 5 geïmplementeerd, maar daarbuiten moeten we de noisy distance 
        # gebruiken met game_state.get_agent_distances, maar noisy distance neemt de manhattan distance en die is niet zo goed
        # want die houdt geen rekening met muren dus idk hoe ik verder moest
        # => Op zich als de ghost meer dan 5 verwijderd is hoeven we er ons helemaal geen zorgen over te maken, pas 
        # als die dichter komt moeten we weglopen. Dus mss hoeven we vr de offensive agent die noisy distance niet eens te gebruiken.
        # Vind ik goeddd!!!!

        # "is Pacman in a tunnel" 0 of 1 aangeven of de actie pacman zal trappen 
        features["dead_end_tunnel"] = 0

        if my_pos in self.dead_end_tunnels:
            entrance = self.dead_end_tunnels[my_pos]

            pacman_dist_to_entrance = self.get_maze_distance(my_pos, entrance)
            current_dist_to_entrance = self.get_maze_distance(current_pos, entrance)
            escaping = pacman_dist_to_entrance < current_dist_to_entrance
            
            for opponent in opponents_list:
                if opponent.is_pacman or opponent.scared_timer > 0:
                    continue # We gaan verder met de volgende opponent in de list want deze opponent kan ons niet aanvallen

                ghost_pos = opponent.get_position() # Is de ghost dichtbij genoeg?
                if ghost_pos is None: # Als het None is, weten we dat de ghost niet 5 distance is, anders heeft die een coördinaat
                    continue # We gaan verder met de volgende opponent in de list want deze opponent is te ver
                
                ghost_dist_to_entrance = self.get_maze_distance(ghost_pos, entrance)
                
                if ghost_dist_to_entrance <= pacman_dist_to_entrance + 3 and not escaping: # + 3 zodat pacman niet een tunnel ingaat wanneer een ghost hem achtervolgt (we kunnen ipv +3 de depth gebruiken om de buffer preciezer te berekenen)
                    features["dead_end_tunnel"] = 1
                    break # Als 1 ghost ons gaat killen in de tunnel, zijn we dood en is de loop klaar

        # We moeten sws nog een feature toepassen dat onze attacker rondgaat als hij heel de tijd de enemy tegenkomt langs dezelfde kant

        return features

    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        inventory_space = successor.get_agent_state(self.index).num_carrying
        distance_to_home_weight = -inventory_space # Hoe meer food we dragen, hoe meer we een verre afstand van ons eigen veld afstraffen
        # Dus wnr we veel food dragen kiest pacman voor acties die hem terug nr huis brengen => zo kan de food die hij draagt ingecasht worden voor punten
        # We kunnen deze berekening ook doen bij de features om zo de weight constant te houden (vb. -1), is mss beter

        return {'remaining_food': -200, # Minimaliseer resterende food (dus eet food)
                'distance_to_food': -4, # Minimaliseer afstand naar de dichtste food dot
                'distance_to_home': distance_to_home_weight, # Probeer naar huis te gaan wnr je veel food draagt
                #'distance_to_capsule': -2, # Ik krijg het niet echt goed met die capsules, plots staat hij vaak stil enzo.
                #'remaining_capsules': -30, # Dus mss beter voor later houden en eerst gwn simpel beginnen zonder capsules
                'distance_to_ghost': -100, # Hoe dichter de ghost van die 5 distance, hoe negatiever de q-value 
                'distance_to_scared_ghost': -8, # Zodat als hij de kans krijgt, dat hij die wel echt neemt
                'ate_scared_ghost': 500,
                'death': -10000, 
                'dead_end_tunnel': -10000} # Nu weet hij zeker dat dit een slechte state is
    # De weights kunnen wrs nog veel beter !


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0 # Interessanttttttt dyanmische agent?

        # Computes distance to invaders we can see # Ik denk dat dit enkel rekening houdt met invaders die within 5 squares zijn
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        # Hier is het vgm ook belangrijk om naar enemies te kijken die verder zijn (dus met noisy distance)

        # We kunnen hier nog een feature toevoegen dat ipv dat de agent random rondloopt, kan hij de plaats met meeste food bewaken
        food_to_defend = self.get_food_you_are_defending(game_state).as_list()
        dists = [self.get_maze_distance(my_pos, food_pos) for food_pos in food_to_defend]
        features['sum_distance_to_food'] = sum(dists)

        capsules_to_defend = self.get_capsules_you_are_defending(successor)
        if len(capsules_to_defend) > 0:
            features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules_to_defend])

        
        # Deze feature is om zo dicht mogelijk bij de grens van rood en blauw te blijven.
        # Ofwel is de pacman al geïnfiltreerd, maar dan moet hij nog terug om punten te verdienen.
        # Als we bij de grens op hem wachten kunnen we hem makkelijk vangen.
        # Ofwel moet hij ons territorium nog binnenkomen. Dan kunnen we hem meteen vangen.
        # Ik heb deze getest en heb het gevoel dat da echt nog goe kan zijn.
        # Ook voor een meer dynamische agent is da mss goed want dan is hij al dicht bij het territorium van de tegenstander
        # en kan hij af en toe food gaan stelen mss
        # ECHT SLIMM!!!!

        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        middle = width // 2
        if self.red:
            middle = middle - 1 # middle is het einde van ons eigen territorium, dus waar pacman terug een spookje wordt
        #else: middle = middle + 1 
        middle_positions = [(middle, h) for h in range(height) if not game_state.has_wall(middle, h)] # Als je maze distance berekent met een positie waar een wall is wordt blijkbaar een error gethrowd
        # number_of_entry_points = len(middle_positions)   
        distances_to_home = [self.get_maze_distance(my_pos, pos) for pos in middle_positions] 
        features['distance_to_closest_boundary'] = min(distances_to_home) # / number_of_entry_points 
        # Die sum is misschien te veel berekeningen doen wanneer we maar 1 seconde compute tijd hebben
        # Als er minder entry points zijn in ons gebied is het meer de moeite waard om daar dicht bij te blijven. Maar dat maakt het nogal ingewikkeld.

        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            if my_state.scared_timer > 0:
                features['invader_distance_while_scared'] = 1 / min(dists) 
            else:
                features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000,
                'on_defense': 100,
                'invader_distance': -20,
                'invader_distance_while_scared': -100,
                'stop': -2, # Ik heb de weight voor stop veel lager gezet want vgm is da voor de defensieve niet zo erg om te stil te staan
                'reverse': -2,
                'sum_distance_to_food': -0.1,
                'distance_to_capsule': -1,
                'distance_to_closest_boundary': -0.1} 
