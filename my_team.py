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

        # We storen alle dead-ends in de layout. Dit wordt dus eenmaal in het begin van het spel berekend.
        self.dead_end_tunnels = {} # Mapt elk coördinaat deel van een dead-end met de ingang van de dead-end, en de diepte tussen de coördinaat en de ingang.

        walls = game_state.get_walls()
        width, height = walls.width, walls.height

        def get_free_neighbours(pos):
            """Returnt alle naburige coördinaten die geen muur zijn"""
            x, y = pos
            neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            return [pos for pos in neighbours if not game_state.has_wall(pos[0], pos[1])]

        free_neighbours = {} # Mapt elk coördinaat met het aantal vrije buren dat nog niet gecheckt is.
        for x in range(width):
            for y in range(height):
                if not game_state.has_wall(x, y):
                    free_neighbours[(x, y)] = len(get_free_neighbours((x, y)))
        
        queue = [pos for pos, n in free_neighbours.items() if n == 1] # Beginnen met uiterste dead-ends:
        # Een positie die maar 1 vrije buur heeft is sowieso een dead-end.
        to_exit = {} # Mapt elk coördinaat die in een dead-end zit met de buur waar we naartoe moeten om uit de dead-end te geraken.
        # Op deze manier creëren we een pad van het einde van de dead-end naar de uitgang.

        while len(queue) > 0: 
            curr = queue.pop()
            active_neighbours = [pos for pos in get_free_neighbours(curr) if free_neighbours[pos] > 0] # Alle buren die nog niet in de dead-end voorkwamen
            if len(active_neighbours) == 0: 
                continue
                
            parent = active_neighbours[0]
            to_exit[curr] = parent # Een weg naar de uitgang
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
            depth = len(path)
            self.dead_end_tunnels[coord] = (entrance, depth)


        self.last_defended_food = self.get_food_you_are_defending(game_state).as_list()
        self.missing_food_dot = None # Houdt de food bij die net is opgegeten door de tegenstander.

    def choose_action(self, game_state):

        current_defended_food = self.get_food_you_are_defending(game_state).as_list()
        if len(self.last_defended_food) > len(current_defended_food):
            missing_food = list(set(self.last_defended_food) - set(current_defended_food)) # Lijst van food die ondertussen door de tegenstander is opgegeten
            if len(missing_food) > 0:
                self.missing_food_dot = missing_food[0] # Dit laat ons toe te tracken waar de tegenstander voor het laatst food heeft opgegeten
        self.last_defended_food = current_defended_food 
    
        my_pos = game_state.get_agent_state(self.index).get_position()
        if self.missing_food_dot == my_pos: # Als we geen enemy vinden bij de missing food dot dan hoeven we er gen rekening meer mee te houden
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

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
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


class OffensiveReflexAgent(ReflexCaptureAgent):

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_dots = self.get_food(successor).as_list()
        capsules = self.get_capsules(successor)
        opponents = [successor.get_agent_state(opponent) for opponent in self.get_opponents(successor)]
        agent = successor.get_agent_state(self.index)
        my_pos = agent.get_position()

        ghosts_within_5 = [ # Houdt enemy ghosts bij op een maze distance van 5 of minder
            ghost 
            for ghost in opponents
            if not ghost.is_pacman and
            ghost.get_position() is not None and 
            ghost.scared_timer == 0 and
            self.get_maze_distance(my_pos, ghost.get_position()) <= 5
        ]
        scared_ghosts = [
            ghost 
            for ghost in opponents
            if not ghost.is_pacman and
            ghost.get_position() is not None and 
            ghost.scared_timer > 0
        ]

        # Offensive Feature 1: eet food dots
        features['remaining_food'] = len(food_dots)  

        # Offensive Feature 2: blijf dicht bij eigen territorium
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        middle = width // 2 # middle is het einde van ons eigen territorium, dus waar pacman terug een spookje wordt
        if self.red:
            middle = middle - 1 
        middle_positions = [(middle, h) for h in range(1, height) if not game_state.has_wall(middle, h)]
        distances_to_home = [self.get_maze_distance(my_pos, pos) for pos in middle_positions] # Kan ook met A*, maar neemt (te) veel compute
        closest_border_point = middle_positions[distances_to_home.index(min(distances_to_home))]
        features['distance_to_home'] = self.get_a_star_distance(my_pos, closest_border_point, successor) # Kortste pad naar border terwijl we de ghost vermijden met A*

        # Offensive Feature 3: verklein afstand naar dichtstbijzijnde food dot
        if len(food_dots) > 0:  # This should always be True,  but better safe than sorry
            min_food_distance = min([self.get_a_star_distance(my_pos, food, successor) for food in food_dots])
            features['distance_to_food'] = min_food_distance

        # Offensive Feature 4: verklein afstand naar dichtstbijzijnde capsule
        if len(capsules) > 0:
            min_capsule_distance = min([self.get_a_star_distance(my_pos, capsule, successor) for capsule in capsules])
            features['distance_to_capsule'] = min_capsule_distance
        # Offensive Feature 5: eet capsules
        features['remaining_capsules'] = len(capsules)

        # Offensive Feature 6: Ga niet dood
        current_pos = game_state.get_agent_state(self.index).get_position()
        features['death'] = 0
        if my_pos == self.start and current_pos != self.start:
            features['death'] = 1
        
        # Offensive Feature 7: blijf weg van (niet scared) spookjes die in de buurt zijn
        if len(ghosts_within_5) > 0 and features['death'] == 0: # Weglopen hoeft niet als we sowieso doodgaan.
            min_ghost_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts_within_5])
            features['distance_to_ghost'] = 1 / min_ghost_distance 
            # We combineren 1 / min_ghost_distance met een negatieve reward.
            # Bij gewoon min_ghost_distance met een positieve reward zou pacman altijd binnen een afstand van 5 van een spook proberen blijven, maar dit is niet het gedrag dat we willen.

        # Offensive feature 8 en 9: probeer toch enemies te killen als je nog een ghost bent (en dus onderweg naar het territorium van de tegenstander)
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


        # Offensive feature 10: verklein afstand met dichtsbijzijnde scared ghost
        if len(scared_ghosts) > 0:
            min_scared_ghost_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in scared_ghosts])
            closest_scared_ghost = [ghost for ghost in scared_ghosts 
                                    if self.get_maze_distance(my_pos, ghost.get_position()) == min_scared_ghost_distance][0]
            if closest_scared_ghost.scared_timer > min_scared_ghost_distance + 2: # We moeten genoeg tijd hebben om de ghost te kunnen catchen
                features['distance_to_scared_ghost'] = min_scared_ghost_distance
            else: # Loop toch weg als de ghost bijna niet meer scared is
                features['distance_to_ghost'] = max(features['distance_to_ghost'], 1 / min_scared_ghost_distance)
        
        # Offensive feature 10: eet de scared ghost
        features['ate_scared_ghost'] = 0
        current_opponents = [game_state.get_agent_state(opponent) for opponent in self.get_opponents(game_state)]
        for ghost in current_opponents:
            if not ghost.is_pacman and ghost.get_position() is not None and ghost.scared_timer > 0:
                if my_pos == ghost.get_position():
                    features['ate_scared_ghost'] = 1

        # Offensive feature 11: ga richting huis indien er weinig tijd is en je nog food dots draagt
        features['final_sprint'] = 0
        current_agent_state = game_state.get_agent_state(self.index)
        if agent.is_pacman and current_agent_state.num_carrying > 0:
            moves_left = game_state.data.timeleft // 4 # Kon geen getter vinden voor deze gamestatedata, // 4 omdat het totale tijd is van alle 4 agents
            min_home_dist = min(distances_to_home)
            if moves_left <= min_home_dist + 5: # +5 tijd geven voor als er een ghost in de weg zit naar huis
                features['final_sprint'] = min_home_dist
            
        # Offensive Feature 12: loop jezelf niet dood in een tunnel
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
                    continue # We gaan verder met de volgende opponent in de list want deze opponent is waarschijnlijk te ver
                ghost_dist_to_entrance = self.get_maze_distance(ghost_pos, entrance)
                if ghost_dist_to_entrance <= successor_dist_to_entrance + depth and not escaping: # + depth zodat pacman niet een tunnel ingaat wanneer een ghost hem achtervolgt
                    features["dead_end_tunnel"] = 1
                    break # Als 1 ghost ons gaat killen in de tunnel, zijn we dood en is de loop klaar

        # Offensive Feature 13: blijf niet stilstaan
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        # Offensive Feature 14: ga niet terug van waar je komt (vermijdt dat pacman heel de tijd gewoon heen en weer gaat)
        if action == rev: features['reverse'] = 1
        return features
    


    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        food_carrying = successor.get_agent_state(self.index).num_carrying
        food_dots = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        my_pos = successor.get_agent_state(self.index).get_position()
 
        # Dynamische weight berekening voor feature 2:
        opponents = [successor.get_agent_state(opponent) for opponent in self.get_opponents(successor)]
        ghosts_within_5 = [
            ghost 
            for ghost in opponents
            if not ghost.is_pacman and
            ghost.get_position() is not None and 
            ghost.scared_timer == 0 and
            self.get_maze_distance(my_pos, ghost.get_position()) <= 5 # muren in rekening houden
        ]
        if len(ghosts_within_5) > 0 and food_carrying > 0:
            distance_to_home_weight = -200 -(food_carrying * 10) # Als we food hebben en ghost is dichtbij, ga snel naar huis
        elif food_carrying >= len(food_dots) - 2: # Naar huis gaan wanneer we max food hebben om te winnen
            distance_to_home_weight = -5 # Net iets meer dan distance_to_food 
        else: 
            distance_to_home_weight = 0 # Niemand in de buurt, dus zoveel mogelijk eten
        
        # Dynamische weight berekening voor features 13, 14, 4 en 2
        stop_weight = -2
        reverse_weight = -3
        capsule_weight = -2
        if len(ghosts_within_5) > 0:
            stop_weight = -100
            reverse_weight = -50 # zodat hij een manier vindt om rond te gaan wanneer hij blijft cirkelen met de enemy bij de border
            if len(capsules) > 0:
                min_ghost_dist = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts_within_5])
                min_capsule_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
                if min_capsule_dist <= min_ghost_dist + 1: # Als we naar de capsule kunnen gaan voor de ghost
                    capsule_weight = -150       # Pak de capsule snel
                    distance_to_home_weight = 0 # ipv naar huis te gaan

        return {'remaining_food': -200,
                'distance_to_food': -4,
                'distance_to_home': distance_to_home_weight,
                'distance_to_capsule': capsule_weight,
                'remaining_capsules': -301,
                'death': -10000, 
                'invader_distance': -8,
                'ate_invader': 500,
                'distance_to_ghost': -100,
                'distance_to_scared_ghost': -15,
                'ate_scared_ghost': 1000,
                'final_sprint': -1000,
                'dead_end_tunnel': -10001,
                'stop': stop_weight, 
                'reverse': reverse_weight}


class DefensiveReflexAgent(ReflexCaptureAgent):

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Defensive Feature 1: elimineer invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        # Hier is het vgm ook belangrijk om naar enemies te kijken die verder zijn (dus met noisy distance)

        # Defensive Feature 2: blijf in het gebied waar de food het meest geconcentreerd is
        food_to_defend = self.get_food_you_are_defending(game_state).as_list()
        dists = [self.get_maze_distance(my_pos, food_pos) for food_pos in food_to_defend]
        features['sum_distance_to_food'] = sum(dists)

        # Defensive Feature 3: minimiseer afstand naar dichtstbijzijnde capsule
        capsules_to_defend = self.get_capsules_you_are_defending(successor)
        if len(capsules_to_defend) > 0:
            features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules_to_defend])

        # Defensive Feature 4: blijf dicht bij de border van de twee territoria
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        middle = width // 2
        if self.red:
            middle = middle - 1 # middle is het einde van ons eigen territorium, dus waar pacman terug een spookje wordt 
        middle_positions = [(middle, h) for h in range(height) if not game_state.has_wall(middle, h)]
        distances_to_home = [self.get_maze_distance(my_pos, pos) for pos in middle_positions] 
        features['distance_to_closest_boundary'] = min(distances_to_home)

        # Defensive Feature 5: Als je even langs enemy territory moet, ga zo snel mogelijk naar huis
        features['enemy_territory_depth'] = 0
        if my_state.is_pacman:
            features['enemy_territory_depth'] = min(distances_to_home)

        # Defensive Feature 6, 7, 8, 9: bepalen de afstand tot de invader afhankelijk van bvb. de scared state
        if len(invaders) > 0:
            self.missing_food_dot = None # Je weet al exact waar de invaders zijn dus missing food dot is niet van belang
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            min_dist = min(dists)
            if my_state.scared_timer > 0:
                if my_state.scared_timer <= 2: # Als de scared timer bijna op is, willen we de enemy killen, 
                    features['invader_distance_while_scared'] = abs(min_dist - 1) # dus probeer op afstand 1 te blijven
                else: # Anders blijven we op 3 afstand om hem daarna zo snel mogelijk te kunnen killen
                    features['invader_distance_while_scared'] = abs(min_dist - 3)
            else:
                features['invader_distance'] = min_dist # Niet scared dus probeer hem meteen te catchen
        else:
            if self.missing_food_dot is not None: # De missing food dot geeft een clue van waar de tegenstander kan zijn
                features['distance_to_missing_food'] = self.get_maze_distance(my_pos, self.missing_food_dot)


        # Defensive Feature 10: eet invaders op
        features['ate_invader'] = 0
        current_opponents = [game_state.get_agent_state(opponent) for opponent in self.get_opponents(game_state)]
        if not my_state.is_pacman and my_state.scared_timer == 0:
            for invader in current_opponents:
                if invader.is_pacman and invader.get_position() is not None:
                    if my_pos == invader.get_position():
                        features['ate_invader'] = 1

        # Defensive Feature 11: vermijd dead-ends als je scared bent
        features["dead_end_tunnel"] = 0 # dead-end feature toegevoegd voor onze scared defender
        current_pos = game_state.get_agent_state(self.index).get_position()
        if my_state.scared_timer > 0 and my_pos in self.dead_end_tunnels:
            entrance, depth = self.dead_end_tunnels[my_pos]
            successor_dist_to_entrance = self.get_maze_distance(my_pos, entrance)
            current_dist_to_entrance = self.get_maze_distance(current_pos, entrance)
            escaping = successor_dist_to_entrance < current_dist_to_entrance
            for invader in invaders:
                invader_pos = invader.get_position()
                if invader_pos is None: # invader is waarschijnlijk ver genoeg
                    continue     
                invader_dist_to_entrance = self.get_maze_distance(invader_pos, entrance)
                if invader_dist_to_entrance <= successor_dist_to_entrance + depth and not escaping: 
                    features["dead_end_tunnel"] = 1
                    break

        # Defensive Feature 12: vermijd stilstaan
        if action == Directions.STOP: features['stop'] = 1
        # Defensive Feature 13: vermijd reverse
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000,
                'enemy_territory_depth': -2,
                'invader_distance': -20,
                'invader_distance_while_scared': -100,
                'distance_to_missing_food': -5,
                'ate_invader': 500,
                'stop': -2, 
                'reverse': -2,
                'sum_distance_to_food': -0.1,
                'distance_to_capsule': -0.2,
                'distance_to_closest_boundary': -1,
                'dead_end_tunnel': -10000,
                } 
    
class DynamicReflexAgent(ReflexCaptureAgent):
    # Klasse variabelen (gemeenschappelijk voor alle instanties)
    shared_roles = {}   # { agent_index: 'offensive' | 'defensive' }
    team_indices = []   # [first_index, second_index], geïnitialiseerd in register_initial_state

    def register_initial_state(self, game_state):
        if len(DynamicReflexAgent.team_indices) >= 2: # Als we meerdere games met 1 python commando runnen, moeten we de team_indices lijst telkens resetten
            DynamicReflexAgent.team_indices = [] 
            DynamicReflexAgent.shared_roles = {} 
        super().register_initial_state(game_state)

        self.prev_pos = self.start # Houd vorige positie bij

        DynamicReflexAgent.team_indices.append(self.index) # Registreer jezelf in de index list

        # Geef de rollen:
        # eerste agent → offensive, tweede → defensive
        if len(DynamicReflexAgent.shared_roles) == 0:
            DynamicReflexAgent.shared_roles[self.index] = 'offensive'
        else:
            DynamicReflexAgent.shared_roles[self.index] = 'defensive'


    def get_teammate_index(self):
        for idx in DynamicReflexAgent.team_indices:
            if idx != self.index:
                return idx

    # Swapt de rollen. Als alternatief konden we een methode schrijven die voor 1 agent de rol verandert zodat ze bvb. allebei offensive worden.
    def swap_roles(self):
        """Called wanneer de agent sterft. Deze wordt dan defensive; de teammate wordt offensive."""
        DynamicReflexAgent.shared_roles[self.index] = 'defensive'
        teammate = self.get_teammate_index()
        DynamicReflexAgent.shared_roles[teammate] = 'offensive'

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        # Detecteer of de agent sterft
        if (my_pos == self.start and
            self.prev_pos != self.start and
            DynamicReflexAgent.shared_roles.get(self.index) == 'offensive'):
            self.swap_roles() # indien ja, swap roles
        self.prev_pos = my_pos

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
