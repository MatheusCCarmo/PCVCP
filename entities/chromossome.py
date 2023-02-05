from aux_functions import *

class Chromossome:

    def __init__(self, route, G):
        self.route = route
        
        self.cost = route_cost(self.route, G)

        self.bonus_colected = calculate_bonus_colected(self.route, G)     


    def fitness_value(self):
        return  1 / self.cost


    