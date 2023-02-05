import math
from entities.vertex import Vertex

calculations_counter = 0

def eucl_dist(x1,y1,x2,y2):
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2 )


def swap_2(i, j, route):
    route_A = route[:i]
    route_B = route[i:j]
    route_B.reverse()
    new_route = [*route_A,*route_B, *route[j:]]
    return new_route

def calculate_route_distance(route, G):

    distance = 0
    # print(route)
    for v in range(len(route)):
        i = route[v - 1]['id']
        j = route[v]['id']        
        distance += G.edges[i,j]['weight']
    return distance


def calculate_penalties(route, G):

    penalties = 0
    for i in range(len(G.nodes)):
        if G.nodes[i] not in route:            
            penalties += G.nodes[i]['penalty']
        
    return penalties

def calculate_bonus_colected(route, G):

    bonus = 0
    for r in route:        
        bonus += G.nodes[r['id']]['bonus']

    return bonus


def bonus_labels(route):
    bonus_label = {}
    for i in range(len(route)):
        bonus_label[route[i]['id']] = route[i]['bonus']
    return bonus_label


def load_dataset(file_name):
    n = 0
    quota = 0
    matrix = []
    dataset = []
    with open(file_name, "r") as f:
        for line in f:
            new_line = line.strip()  # remove spaces at the beginning and the end if they are available
            
            new_line = new_line.split(" ")  # split a string into a list
            new_line = list(filter(lambda x: x != '', new_line))
    
            if(len(new_line)==0):
                continue

            if(n == 0):
                n = int(new_line[0])  # check dataset file to see why id,x,y = 0,1,2
                continue


            if(len(matrix) < n):
                matrix.append(new_line)          
                continue

            if(len(new_line)==3):
                id, bonus = new_line[0], int(new_line[1])  # check dataset file to see why id,x,y = 0,1,2
                dataset.append(Vertex(id=id, bonus=bonus))
                continue

            if(len(new_line) == 1):
                quota = int(new_line[0])
                continue

            

        
    return n, quota, matrix, dataset

def load_symmetric(file_name):
    dataset = []
    is_coord = False
    with open(file_name, "r") as f:
        for line in f:
            new_line = line.strip()  # remove spaces at the beginning and the end if they are available
            if(new_line == 'NODE_COORD_SECTION' or new_line == 'DISPLAY_DATA_SECTION'):
                is_coord = True
                continue
            elif(new_line == 'EOF'):
                break
            if(is_coord):
                new_line = new_line.split(" ")  # split a string into a list
                new_line = list(filter(lambda x: x != '', new_line))
                id, x, y = new_line[0], new_line[1], new_line[2]  # check dataset file to see why id,x,y = 0,1,2
                dataset.append(Vertex(id=id, x=x, y=y))  # Create a Node object with id, x, y and add to the data list
    return dataset

def load_asymmetric(file_name):
    dataset = []
    is_coord = False
    with open(file_name, "r") as f:
        for line in f:
            new_line = line.strip()  # remove spaces at the beginning and the end if they are available
            if(new_line == 'NODE_COORD_SECTION' or new_line == 'DISPLAY_DATA_SECTION'):
                is_coord = True
                continue
            elif(new_line == 'EOF'):
                break
            if(is_coord):
                new_line = new_line.split(" ")  # split a string into a list
                new_line = list(filter(lambda x: x != '', new_line))
                id, x, y = new_line[0], new_line[1], new_line[2]  # check dataset file to see why id,x,y = 0,1,2
                dataset.append(Vertex(id=id, x=x, y=y))  # Create a Node object with id, x, y and add to the data list
    return dataset


def route_cost(route, G):
    route_cost.counter += 1
    penalties = calculate_penalties(route, G)
    distance = calculate_route_distance(route, G)
    cost = penalties + distance
    return cost

route_cost.counter = 0