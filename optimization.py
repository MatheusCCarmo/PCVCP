from ortools.linear_solver import pywraplp
import itertools

def solve_pctsp(node_prizes, node_penalties, matrix, quota):

    # Initialize the solver
    solver = pywraplp.Solver(
        'PCTSP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    solver.set_time_limit(100000)

    # Create variables
    n = len(node_prizes)
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = solver.BoolVar(f'x[{i},{j}]')

    y = {}
    for i in range(n):
        y[i] = solver.BoolVar(f'y[{i}]')

    # Set the objective function
    objective = solver.Objective()
    for i in range(n):
        for j in range(n):
            if (i != j):
                objective.SetCoefficient(x[i, j], matrix[i][j])
        objective.SetCoefficient(y[i], -node_penalties[i])
    objective.SetOffset(sum(node_penalties))
    objective.SetMinimization()

    # Add constraints
    for i in range(n):
        out_constr = solver.Constraint(0, 0)
        in_constr = solver.Constraint(0, 0)
        for j in range(n):
            if i != j:
                out_constr.SetCoefficient(x[i, j], 1)
                in_constr.SetCoefficient(x[j, i], 1)
        out_constr.SetCoefficient(y[i], -1)
        in_constr.SetCoefficient(y[i], -1)

    for i in range(n):
        for j in range(i+1, n):
            constr = solver.Constraint(0, 1)
            constr.SetCoefficient(x[i, j], 1)
            constr.SetCoefficient(x[j, i], 1)

    # Add subtour elimination constraint
    for k in range(2, n):
        for subset in itertools.combinations(range(1, n), k):
            s = [0] + list(subset)
            constr = solver.Constraint(-solver.infinity(), k - 1)
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    constr.SetCoefficient(x[s[i], s[j]], 1)

    constraint = solver.Constraint(quota, solver.infinity(), 'minimum_prize')
    for i in range(n):
        constraint.SetCoefficient(y[i], node_prizes[i])

    # Solve the problem
    status = solver.Solve()

    # Print the solution as a formulation
    # print('Objective:', objective.Value())
    # for i in range(n):
    #     for j in range(n):
    #         if (i != j):
    #             print(f'x[{i},{j}] = {x[i, j].solution_value()}')
    # for i in range(n):
    #     print(f'y[{i}] = {y[i].solution_value()}')

    # Extract solution
    if status == pywraplp.Solver.OPTIMAL:
        # Extract solution
        tour = []
        i = 0
        while len(tour) < n and sum(node_prizes[t] for t in tour) < quota:
            tour.append(i)
            for j in range(n):
                if x[i, j].solution_value() >= 0.5 and j not in tour:
                    i = j
                    break
        cost = objective.Value()
        return tour, cost
    else:
        return None, None