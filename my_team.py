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
import time
import math

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DynamicReflexAgent', second='DynamicReflexAgent', num_training=0):
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

        free_neighbours = {}
        for x in range(width):
            for y in range(height):
                if not game_state.has_wall(x, y):
                    free_neighbours[(x, y)] = len(get_free_neighbours((x, y)))
        
        queue = [pos for pos, n in free_neighbours.items() if n == 1] # Beginnen met uiterste dead-ends

        to_exit = {}

        while len(queue) > 0: 
            curr = queue.pop() # Neem de dead-ends stuk voor stuk
            active_neighbours = [n for n in get_free_neighbours(curr) if free_neighbours[n] > 0]
            if len(active_neighbours) == 0: # Zoek de neighbour die nog niet gecheckt is
                continue
                
            parent = active_neighbours[0] # Weg naar de uitgang
            to_exit[curr] = parent
            free_neighbours[curr] = 0 
            free_neighbours[parent] -= 1 # Doe dit weg

            if free_neighbours[parent] == 1: # Als neighbour/parent ook dead-end is geworden, voeg toe aan queue
                    queue.append(parent)
            
        for coord in to_exit.keys(): 
            curr = coord
            path = []
                
            while curr in to_exit: # Volg het pad tot we bij een non-dead end aankomen
                path.append(curr)
                curr = to_exit[curr]
                    
            entrance = curr
            depth = len(path) # Lengte van pad is diepte van de entrance die meerdere tunnels bevat
            self.dead_end_tunnels[coord] = (entrance, depth)


        self.last_defended_food = self.get_food_you_are_defending(game_state).as_list()
        self.missing_food_dot = None

    def choose_action(self, game_state):
        current_defended_food = self.get_food_you_are_defending(game_state).as_list()
        if len(self.last_defended_food) > len(current_defended_food):
            missing_food = list(set(self.last_defended_food) - set(current_defended_food)) # Lijst van missing food
            if len(missing_food) > 0:
                self.missing_food_dot = missing_food[0] # Naar eerste missing food dot gaan
        self.last_defended_food = current_defended_food 
    
        my_pos = game_state.get_agent_state(self.index).get_position()
        if self.missing_food_dot == my_pos: # Als er geen enemy is bij missing food dot, dan gaan we terug
            # Als de enemy al gekillt is onderweg, gaat hij nog steeds naar die food dot, maar dat moet niet per se 
            # tenzij ze twee attackers hebben die exact hetzelfde doen ofzo, maar denk dat hij al kan teruggaan als hij een kill heeft gemaakt
            self.missing_food_dot = None
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

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
    
    def get_a_star_distance(self, start_pos, goal_pos, game_state): # A* die straft op hoe dichtbij we bij een ghost zijn
        opponents = [game_state.get_agent_state(opponent) for opponent in self.get_opponents(game_state)]
        ghost_positions = [
            ghost.get_position() 
            for ghost in opponents
            if not ghost.is_pacman and 
            ghost.get_position() is not None and 
            ghost.scared_timer == 0
        ]
        agenda = util.PriorityQueue()
        agenda.push((start_pos, 0), 0)
        already_visited = {start_pos: 0}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # alle mogelijke moves

        while True:
            if agenda.is_empty():
                return []

            current_pos, current_cost = agenda.pop()

            if current_pos == goal_pos:
                return current_cost

            for dx, dy in directions:
                next_pos = (int(current_pos[0] + dx), int(current_pos[1] + dy))
                if not game_state.has_wall(next_pos[0], next_pos[1]):
                    step_cost = 1 
                    for ghost_pos in ghost_positions:
                        dist_to_ghost = self.get_maze_distance(next_pos, ghost_pos)
                        if dist_to_ghost <= 2:
                            step_cost += 1000
                        elif dist_to_ghost == 3:
                            step_cost += 100

                    total_cost = current_cost + step_cost
                    heuristic = util.manhattan_distance(next_pos, goal_pos)
                    estimated_total_cost = total_cost + heuristic

                    if next_pos not in already_visited or total_cost < already_visited[next_pos]:
                            already_visited[next_pos] = total_cost
                            agenda.push((next_pos, total_cost), estimated_total_cost)


class OffensiveReflexAgent(ReflexCaptureAgent): # inspiratie van de slides Approximate Agent (code lijkt daar beetje op)
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_dots = self.get_food(successor).as_list()
        capsules = self.get_capsules(successor)
        opponents = [successor.get_agent_state(opponent) for opponent in self.get_opponents(successor)]
        ghost_within_5 = [
            ghost 
            for ghost in opponents
            if not ghost.is_pacman and
            ghost.get_position() is not None and 
            ghost.scared_timer == 0
        ] # Misschien beter als we zo typen wanneer tekst niet meer op het scherm past?
        scared_ghosts = [
            ghost 
            for ghost in opponents
            if not ghost.is_pacman and
            ghost.get_position() is not None and 
            ghost.scared_timer > 0
        ]

        agent = successor.get_agent_state(self.index)
        my_pos = agent.get_position()

        # Om food op te eten
        features['remaining_food'] = len(food_dots)  # self.get_score(successor)

        width = game_state.data.layout.width
        height = game_state.data.layout.height
        middle = width // 2
        if self.red:
            middle = middle - 1 # middle is het einde van ons eigen territorium, dus waar pacman terug een spookje wordt
        middle_positions = [(middle, h) for h in range(1, height) if not game_state.has_wall(middle, h)] # Als je maze distance berekent met een positie waar een wall is wordt blijkbaar een error gethrowd
        distances_to_home = [self.get_maze_distance(my_pos, pos) for pos in middle_positions] # geen A* want te veel compute
        closest_border_point = middle_positions[distances_to_home.index(min(distances_to_home))]
        features['distance_to_home'] = self.get_a_star_distance(my_pos, closest_border_point, successor) # Kortste pad naar border terwijl we de ghost vermijden met A*

        # Compute distance to the nearest food
        if len(food_dots) > 0:  # This should always be True,  but better safe than sorry
            min_food_distance = min([self.get_a_star_distance(my_pos, food, successor) for food in food_dots])
            features['distance_to_food'] = min_food_distance

        # Compute distance to the nearest capsule (lekker voor als de ghost dichtbij is)
        if len(capsules) > 0:
            min_capsule_distance = min([self.get_a_star_distance(my_pos, capsule, successor) for capsule in capsules])
            #if min_capsule_distance <= 10:
            features['distance_to_capsule'] = min_capsule_distance
            # else: features['distance_to_capsule'] = 11 # Anders zal die altijd verder dan 10 willen blijven omdat er geen negatieve reward is
            # in plaats van die 11 gewoon de weight verlagen van capsule? want is beetje hardcoded
        features['remaining_capsules'] = len(capsules)

        # Doodgaan is heel slechttt (dat wist pacman niet als hij zat in de tunnel, hij begon daar gewoon te chillen)
        current_pos = game_state.get_agent_state(self.index).get_position()
        features['death'] = 0
        if my_pos == self.start and current_pos != self.start:
            features['death'] = 1
        
        # Compute distance to the nearest ghost within Manhattan distance 5
        # Als de ghosts scared zijn, hoeven we er niet van weg te gaan (maar mss als de timer net gedaan gaat zijn wel, kunnen we nog verbeteren)
        if len(ghost_within_5) > 0 and features['death'] == 0: # Weglopen is alleen belangrijk als we levend kunnen blijven (en dus niet vastzitten in een tunnel)
            min_ghost_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghost_within_5])
            features['distance_to_ghost'] = 1 / min_ghost_distance  # 1/ want pacman bij die positieve reward/weight probeerde pacman die 5 distance te houden, 1/ is dus beter met negatieve reward/weight

        # Compute distance to the nearest invader while being a ghost
        # Zag hem al zo vaak de pacman skippen als hij onderweg was en dat moest gefixt wordennn
        features['invader_distance'] = 0
        features['ate_invader'] = 0
        if not agent.is_pacman:
            invaders = [a for a in opponents if a.is_pacman and a.get_position() is not None]
            if len(invaders) > 0:
                min_invader_distance = min([self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders])
                if agent.scared_timer > 0:
                    features['distance_to_ghost'] = max(features.get('distance_to_ghost', 0), 1 / min_invader_distance)
                else:
                    features['invader_distance'] = min_invader_distance

            for opponent in opponents:
                if opponent.is_pacman and opponent.get_position() is not None:
                    if my_pos == opponent.get_position() and agent.scared_timer == 0:
                        features['ate_invader'] = 1


        # Compute distance to the nearest scared ghost
        if len(scared_ghosts) > 0:
            min_scared_ghost_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in scared_ghosts])
            closest_scared_ghost = [ghost for ghost in scared_ghosts if self.get_maze_distance(my_pos, ghost.get_position()) == min_scared_ghost_distance][0]
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

        features['final_sprint'] = 0
        current_agent_state = game_state.get_agent_state(self.index)
        if agent.is_pacman and current_agent_state.num_carrying > 0:
            moves_left = game_state.data.timeleft // 4 # Kon geen getter vinden voor deze gamestatedata, // 4 omdat het totale tijd is van alle 4 agents
            min_home_dist = min(distances_to_home)
            if moves_left <= min_home_dist + 5: # +5 tijd geven voor als er een ghost in de weg zit naar huis
                features['final_sprint'] = min_home_dist
            
        # Nu alleen nog maar de distance binnen die 5 geïmplementeerd, maar daarbuiten moeten we de noisy distance 
        # gebruiken met game_state.get_agent_distances, maar noisy distance neemt de manhattan distance en die is niet zo goed
        # want die houdt geen rekening met muren dus idk hoe ik verder moest
        # => Op zich als de ghost meer dan 5 verwijderd is hoeven we er ons helemaal geen zorgen over te maken, pas 
        # als die dichter komt moeten we weglopen. Dus mss hoeven we vr de offensive agent die noisy distance niet eens te gebruiken.
        # Vind ik goeddd!!!!

        # "is Pacman in a tunnel" 0 of 1 aangeven of de actie pacman zal trappen 
        features["dead_end_tunnel"] = 0

        if my_pos in self.dead_end_tunnels:
            entrance, depth = self.dead_end_tunnels[my_pos]

            successor_dist_to_entrance = self.get_maze_distance(my_pos, entrance)
            current_dist_to_entrance = self.get_maze_distance(current_pos, entrance)
            escaping = successor_dist_to_entrance < current_dist_to_entrance
            
            for opponent in opponents:
                if opponent.is_pacman or opponent.scared_timer > 0:
                    continue # We gaan verder met de volgende opponent in de list want deze opponent kan ons niet aanvallen

                ghost_pos = opponent.get_position() # Is de ghost dichtbij genoeg?
                if ghost_pos is None: # Als het None is, weten we dat de ghost niet 5 distance is, anders heeft die een coördinaat
                    continue # We gaan verder met de volgende opponent in de list want deze opponent is te ver
                
                ghost_dist_to_entrance = self.get_maze_distance(ghost_pos, entrance)
                if ghost_dist_to_entrance <= successor_dist_to_entrance + depth and not escaping: # + depth zodat pacman niet een tunnel ingaat wanneer een ghost hem achtervolgt
                    features["dead_end_tunnel"] = 1
                    break # Als 1 ghost ons gaat killen in de tunnel, zijn we dood en is de loop klaar
        
        # Niet kunnen testen met domme baseline :') dit wordt hem niet, denk dat we dit beter weglaten
        '''features['teammate_distance_penalty'] = 0 # Zodat onze twee attackers niet dezelfde richting op gaan
        teammates = [game_state.get_agent_state(i) for i in self.get_team(successor) if i != self.index]
        
        for teammate in teammates:
            if teammate.get_position() is not None:
                dist = self.get_maze_distance(my_pos, teammate.get_position())
                if dist <= 3:
                    features['teammate_distance_penalty'] = 1'''


        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        # ^^^
        # We moeten sws nog een feature toepassen dat onze attacker rondgaat als hij heel de tijd de enemy tegenkomt langs dezelfde kant
        return features

    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        food_carrying = successor.get_agent_state(self.index).num_carrying
        food_dots = self.get_food(game_state).as_list()
 
        # Dit hieronder toegevoegd zodat als hij een kill maakt, hij niet instant naar huis gaat om zijn food binnen te brengen,
        # maar eerst nog zoveel mogelijk food gaat halen
        opponents = [successor.get_agent_state(opponent) for opponent in self.get_opponents(successor)]
        ghost_within_5 = [
            ghost 
            for ghost in opponents
            if not ghost.is_pacman and
            ghost.get_position() is not None and 
            ghost.scared_timer == 0
        ]
        if len(ghost_within_5) > 0 and food_carrying > 0:
            distance_to_home_weight = -200 -(food_carrying * 10) # Als we food hebben en ghost is dichtbij, snel naar huis
        elif food_carrying >= len(food_dots) - 2: # Naar huis gaan wanneer we max food hebben om te winnen
            distance_to_home_weight = -5 # Net iets meer dan distance_to_food 
        else: 
            distance_to_home_weight = 0 # Niemand in de buurt, dus zoveel mogelijk eten
        
        stop_weight = -2
        reverse_weight = -3
        if len(ghost_within_5) > 0:
            stop_weight = -100
            reverse_weight = -50 # zodat hij een manier vindt om rond te gaan wanneer hij blijft cirkelen met de enemy bij de border

        return {'remaining_food': -200, # Minimaliseer resterende food (dus eet food)
                'distance_to_food': -4, # Minimaliseer afstand naar de dichtste food dot
                'distance_to_home': distance_to_home_weight, # Probeer naar huis te gaan wnr je veel food draagt
                'distance_to_capsule': -2,
                'remaining_capsules': -301,
                'invader_distance': -8,
                'ate_invader': 500,
                'distance_to_ghost': -100, # Hoe dichter de ghost van die 5 distance, hoe negatiever de q-value 
                'distance_to_scared_ghost': -15, # Zodat als hij de kans krijgt, dat hij die wel echt neemt
                'ate_scared_ghost': 500,
                #'teammate_distance_penalty': -15,
                'final_sprint': -10000,
                'death': -10000, 
                'dead_end_tunnel': -10001, # Nu weet hij zeker dat dit een slechte state is
                'stop': stop_weight, 
                'reverse': reverse_weight}
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
        # features['on_defense'] = 1
        # if my_state.is_pacman: features['on_defense'] = 0 # Interessanttttttt dyanmische agent?

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
        middle_positions = [(middle, h) for h in range(height) if not game_state.has_wall(middle, h)] # Als je maze distance berekent met een positie waar een wall is wordt blijkbaar een error gethrowd
        # number_of_entry_points = len(middle_positions)   
        distances_to_home = [self.get_maze_distance(my_pos, pos) for pos in middle_positions] 
        features['distance_to_closest_boundary'] = min(distances_to_home) # / number_of_entry_points 
        # Die sum is misschien te veel berekeningen doen wanneer we maar 1 seconde compute tijd hebben
        # Als er minder entry points zijn in ons gebied is het meer de moeite waard om daar dicht bij te blijven. Maar dat maakt het nogal ingewikkeld.
        # Als er misschien 1 entry point is, kunnen we dat wel echt blijven bewaken !!!

        # Voor die speciale crowdedCapture map kan hij nu snel door enemy territory gaan en een defender blijven
        features['enemy_territory_depth'] = 0
        if my_state.is_pacman:
            features['enemy_territory_depth'] = min(distances_to_home)

        if len(invaders) > 0:
            self.missing_food_dot = None
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            min_dist = min(dists)
            if my_state.scared_timer > 0:
                if my_state.scared_timer <= 2: # Als de scared timer bijna op is, willen we de enemy killen
                    features['invader_distance_while_scared'] = abs(min_dist - 1) # soms ging hij in de enemy vlak voordat hij unscared ging
                else: # Anders blijven we op 3 afstand om hem daarna zo snel mogelijk te kunnen killen
                    features['invader_distance_while_scared'] = abs(min_dist - 3)
            else:
                features['invader_distance'] = min_dist
            '''else: # len(invaders)=0 dus mogelijke invaders zijn op manhattan afstand > 5
            noisy_distances = successor.get_agent_distances()
            opp_indices = self.get_opponents(successor)
            far_invaders = [noisy_distances[i] for i in opp_indices if successor.get_agent_state(i).is_pacman]
            features['far_invader_distance'] = min(noisy_distances)'''
        else:
            if self.missing_food_dot is not None:
                features['distance_to_missing_food'] = self.get_maze_distance(my_pos, self.missing_food_dot)

        features['ate_invader'] = 0
        current_opponents = [game_state.get_agent_state(opponent) for opponent in self.get_opponents(game_state)]
        
        if not my_state.is_pacman and my_state.scared_timer == 0:
            for invader in current_opponents:
                if invader.is_pacman and invader.get_position() is not None:
                    if my_pos == invader.get_position():
                        features['ate_invader'] = 1

        features['wrong_defender_side'] = 0 # Verdelen de twee defenders over de bovenste en onderste helft
        if len(invaders) == 0 and self.missing_food_dot is None:
            my_team = self.get_team(successor)
            top_defender = (self.index == min(my_team))
            middle_height = height // 2
            _, y = my_pos
            
            if top_defender and y < middle_height: # Als ik de top defender ben, maar ik zit op de onderste helft
                features['wrong_defender_side'] = 1
            elif not top_defender and y >= middle_height: # Als ik de bottom defender ben, maar ik zit op de bovenste helft
                features['wrong_defender_side'] = 1

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000,
                # 'on_defense': 100,
                'enemy_territory_depth': -2,
                'invader_distance': -20,
                'invader_distance_while_scared': -100,
                # 'far_invader_distance': -10, # door de noisyness is dit denk ik eig niet zo nuttig, bijna de helft vd tijd geeft dit een verkeerde suggestie
                'distance_to_missing_food': -5,
                'ate_invader': 500,
                'stop': -2, # Ik heb de weight voor stop veel lager gezet want vgm is da voor de defensieve niet zo erg om te stil te staan
                'reverse': -2,
                'sum_distance_to_food': -0.1,
                'distance_to_capsule': -0.2,
                'distance_to_closest_boundary': -1,
                # 'wrong_defender_side': -10
                } 
    
class DynamicReflexAgent(ReflexCaptureAgent):
    # Klasse variabelen (gemeenschappelijk voor alle instanties)
    shared_roles = {}   # { agent_index: 'offensive' | 'defensive' }
    team_indices = []   # [first_index, second_index], geinitialiseerd in register_initial_state

    def register_initial_state(self, game_state):
        if len(DynamicReflexAgent.team_indices) >= 2: # meerdere games worden met 1 python commando gerunt, dus onze team_indices lijst wordt nooit gerest
            DynamicReflexAgent.team_indices = []      # omdat het klassevariabelen zijn, blijven die altijd in memory en bouwt die lijst op
            DynamicReflexAgent.shared_roles = {}      # bv. game 1: team_indices = [0,2]; game 2: team_indices = [0, 2, 0, 2] ...  
        super().register_initial_state(game_state)

        self.prev_pos = self.start # Altijd vorige positie bijhouden

        # Registreer in index list
        DynamicReflexAgent.team_indices.append(self.index)

        # Geef de rollen:
        # eerste agent → offensive, tweede → defensive
        if len(DynamicReflexAgent.shared_roles) == 0:
            DynamicReflexAgent.shared_roles[self.index] = 'offensive'
        else:
            DynamicReflexAgent.shared_roles[self.index] = 'defensive'


    # Kan wrs ook met andere methoden maar dit is nog wel simpel op zich
    def _get_teammate_index(self):
        for idx in DynamicReflexAgent.team_indices:
            if idx != self.index:
                return idx

    # Swapt de rollen. We kunnen ook methode schrijven die gwn voor 1 agent de rol veranderd zodat ze bvb. allebei offensive worden
    def _swap_roles(self):
        """Called wnr de agent sterft. Deze wordt dan defensive; teammate wordt offensive."""
        DynamicReflexAgent.shared_roles[self.index] = 'defensive'
        teammate = self._get_teammate_index()
        DynamicReflexAgent.shared_roles[teammate] = 'offensive'

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Death detection (dood als agent respawnt bij start)
        if (my_pos == self.start and
            self.prev_pos != self.start and
            DynamicReflexAgent.shared_roles.get(self.index) == 'offensive'):
            self._swap_roles()
        self.prev_pos = my_pos

        score = self.get_score(game_state)
        moves_left = game_state.data.timeleft // 4
        food_dots = self.get_food(game_state).as_list()

        # Dit werkt niet goed :C zou het niet meer implementeren eigenlijk
        #if score > len(food_dots) / 2: # Als we winnen met meer dan de helft van het eten, gaan beide agents defensive spelen
        #    DynamicReflexAgent.shared_roles[self.index] = 'defensive'
            
        #if score < 0 and moves_left < 60: # Als we verliezen en de tijd is bijna op, gaan we beide in attack (niet getest)
        #    DynamicReflexAgent.shared_roles[self.index] = 'offensive'

        return super().choose_action(game_state)

    # Nu gebruiken we rechtstreeks onze andere bestaande klassen, maar als we die niet meer gebruiken kan alle code in deze klasse (is conceptueel dan logischer)
    def get_features(self, game_state, action):
        if DynamicReflexAgent.shared_roles.get(self.index) == 'offensive':
            return OffensiveReflexAgent.get_features(self, game_state, action)
        else:
            return DefensiveReflexAgent.get_features(self, game_state, action)

    def get_weights(self, game_state, action):
        if DynamicReflexAgent.shared_roles.get(self.index) == 'offensive':
            return OffensiveReflexAgent.get_weights(self, game_state, action)
        else:
            return DefensiveReflexAgent.get_weights(self, game_state, action)
