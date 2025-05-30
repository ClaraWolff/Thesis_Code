import json
from docplex.mp.model import Model
from docplex.mp.model import Model
from copy import deepcopy
from docplex.mp.model import Model
import json

def compute_route_cost(route, km, index_to_node):
    total_cost = 0
    for i in range(len(route) - 1):
        from_node = index_to_node[route[i]]
        to_node = index_to_node[route[i + 1]]
        total_cost += km[from_node.global_index][to_node.global_index]

    return total_cost

def build_TSP_problem(instance, nodes, times, km, tech, vehicles, test=False, **kwargs):
        customers = [n for n in nodes if n.index not in vehicles and n.index in instance]+[nodes[-1]]

        bigM = 10  # Large constant for time constraints

        # Model Declaration
        m = Model(name='MSVRPTW',**kwargs)
        m.context.cplex_parameters.randomseed=42

        # ----- Variables -----
        m.x_var = m.binary_var_matrix(nodes, nodes, name='route')
        m.t_var = m.continuous_var_dict(nodes, name='arrival_time')

        # ----- Constraints -----
        # Ensure each customer is visited exactly once
        for j in customers:
            m.add_constraint(
                m.sum(m.x_var[i, j] for i in nodes if i != j) == 1,
                ctname=f"visit_customer_{j.index}"
            )

        # Flow conservation constraint
        for i in customers:
                m.add_constraint(
                    m.sum(m.x_var[i, j] for j in nodes if j != i) ==
                    m.sum(m.x_var[j, i] for j in nodes if j != i),
                    ctname=f"flow_conservation_{i.index}"
                )
        
        # Ensure each vehicle starts and ends at its own depot
        m.add_constraint(
            m.sum(m.x_var[tech, j] for j in nodes if j.index != tech.index) ==1,
            ctname=f"start_at_depot_{tech.index}"
        )
        m.add_constraint(
            m.sum(m.x_var[i, tech] for i in nodes if i != tech) == 1,
            ctname=f"return_to_depot_{tech.index}"
        )

        # Time window constraints
        for i in customers: 
            for j in nodes:
                m.add_constraint(
                    m.t_var[j] >= m.t_var[i] + i.serv_time + times[i.global_index][j.global_index] - bigM * (1 - m.x_var[i, j]),
                    ctname=f"time_window_{tech.index}_{i.index}_{j.index}"
                )

        # Enforce time windows
        for i in customers:
            m.add_constraint(
                i.time_window[0] <= m.t_var[i],
                ctname=f"min_time_{i.index}"
            )
            m.add_constraint(
                m.t_var[i] <= i.time_window[1],
                ctname=f"max_time_{i.index}"
            )
        
        # Technician working hours (departure constraint)
        for i in customers:
            m.add_constraint(
                tech.time_window[0] <= m.t_var[i] + bigM * (1 - m.x_var[tech, i]),
                ctname=f"departure_constraint_{i.index}"
            )
    
        # Technician working hours (arrival constraint)
        m.add_constraint(
            m.t_var[tech] <= tech.time_window[1],
            ctname=f"arrival_constraint_tech"
        )


        for i in customers:
            m.add_constraint(
                m.t_var[tech] >= m.t_var[i] + i.serv_time + times[i.global_index][tech.global_index] - bigM * (1 - m.x_var[i, tech]),
                ctname=f"return_to_depot_time_{i.index}"
            )

        # Ensure the first customer visit accounts for depot travel time
        for j in customers:
            m.add_constraint(
                m.t_var[j] >= tech.time_window[0] + times[tech.global_index][j.global_index] - bigM * (1 - m.x_var[tech, j]),
                ctname=f"first_customer_travel_time_{j.index}"
            )


        for i in nodes:
            for j in customers:
                if i != j:  # Avoid self-loops
                    m.add_constraint(
                        km[i.global_index][j.global_index] * m.x_var[i, j] <= 80,  
                        ctname=f"max_travel_distance_{i.index}_{j.index}"
                    )

        non_tech=[v for v in vehicles if v != tech.index]
        for i in nodes:
             for j in nodes:
                if i.index in non_tech or i not in customers+[tech]:
                  m.add_constraint(m.x_var[i,j]==0)
                  m.add_constraint(m.x_var[j,i]==0)
        
        
        # Objective: Minimize total travel cost
        m.minimize(m.sum(km[i.global_index][j.global_index] * m.x_var[i, j] for i in nodes for j in nodes))
        
        return m

def reconstruct_path(vehicle_routes,vehicle):
        path = []
        start=None
        current_node = vehicle.index
        while True:
            next_nodes = [j for (i, j) in vehicle_routes if i == current_node]
            if not next_nodes or current_node==start:
                break
            start=vehicle.index
            current_node = next_nodes[0]
            path.append(current_node)
        return [vehicle.index] + path


def solve_TSP_with_FTR(instance, tech_idx, nodes, new_customer, vehicles, times, km, penalty, lambda_ftr):
  
    bigM = 25
    tech = nodes[tech_idx]
    nodes_with_new = nodes + [new_customer]

    customers = [n for n in nodes_with_new if n.index not in vehicles and n.index in instance] + [new_customer]

    model = Model(name="TSP_with_FTR")
    model.context.cplex_parameters.randomseed=42

    x = model.binary_var_matrix(nodes_with_new, nodes_with_new, name='route')
    t = model.continuous_var_dict(nodes_with_new, name='arrival_time')
    model.x_var = x
    model.t_var = t

    # Customer visit constraints
    for j in customers:
        model.add_constraint(
            model.sum(x[i, j] for i in nodes_with_new if i != j) == 1,
            ctname=f"visit_customer_{j.index}"
        )

    # Flow conservation
    for i in customers:
        model.add_constraint(
            model.sum(x[i, j] for j in nodes_with_new if j != i) ==
            model.sum(x[j, i] for j in nodes_with_new if j != i),
            ctname=f"flow_conservation_{i.index}"
        )

    # Start and return to depot
    model.add_constraint(
        model.sum(x[tech, j] for j in nodes_with_new if j != tech) == 1,
        ctname=f"start_at_depot_{tech.index}"
    )
    model.add_constraint(
        model.sum(x[i, tech] for i in nodes_with_new if i != tech) == 1,
        ctname=f"return_to_depot_{tech.index}"
    )

    # Time constraints
    for i in customers:
        for j in nodes_with_new:
            if i != j:
                model.add_constraint(
                    t[j] >= t[i] + i.serv_time + times[i.global_index][j.global_index] - bigM * (1 - x[i, j]),
                    ctname=f"time_window_{tech.index}_{i.index}_{j.index}"
                )

    for i in customers:
        model.add_constraint(i.time_window[0] <= t[i], ctname=f"min_time_{i.index}")
        model.add_constraint(t[i] <= i.time_window[1], ctname=f"max_time_{i.index}")
        model.add_constraint(
            tech.time_window[0] <= t[i] + bigM * (1 - x[tech, i]),
            ctname=f"departure_constraint_{i.index}"
        )

    model.add_constraint(t[tech] <= tech.time_window[1], ctname="arrival_constraint_tech")

    for i in customers:
        model.add_constraint(
            t[tech] >= t[i] + i.serv_time + times[i.global_index][tech.global_index] - bigM * (1 - x[i, tech]),
            ctname=f"return_to_depot_time_{i.index}"
        )

    for j in customers:
        model.add_constraint(
            t[j] >= tech.time_window[0] + times[tech.global_index][j.global_index] - bigM * (1 - x[tech, j]),
            ctname=f"first_customer_travel_time_{j.index}"
        )

    # Distance constraint
    for i in nodes_with_new:
        for j in customers:
            if i != j:
                model.add_constraint(
                    km[i.global_index][j.global_index] * x[i, j] <= 80,
                    ctname=f"max_travel_distance_{i.index}_{j.index}"
                )

    # Prevent non-tech depots from being used
    non_techs = [v for v in vehicles if v != tech.index]
    for i in nodes_with_new:
        for j in nodes_with_new:
            if i.index in non_techs or i not in customers + [tech]:
                model.add_constraint(x[i, j] == 0)
                model.add_constraint(x[j, i] == 0)

    # FTR penalty
    ftr_score = tech.ftr_by_type.get(new_customer.type, 1)
    ftr_penalty = lambda_ftr * (1 - ftr_score) * penalty
    model.minimize(
        model.sum(km[i.global_index][j.global_index] * x[i, j] for i in nodes_with_new for j in nodes_with_new) + ftr_penalty
    )

    solution = model.solve(time_limit=20)
    if solution is None:
        return float('inf'), [], new_customer, nodes_with_new

    # Extract route
    sol_dict = json.loads(solution.export_as_json_string())
    vehicle_routes = []
    for var in sol_dict['CPLEXSolution'].get('variables', []):
        if var['name'].startswith('route') and float(var['value']) > 0.5:
            parts = var['name'].split('_')
            i = int(parts[1].split('<')[1].split('>')[0])
            j = int(parts[2].split('<')[1].split('>')[0])
            vehicle_routes.append((i, j))

    path = reconstruct_path(vehicle_routes, tech)

    for i in nodes_with_new:
        for j in nodes_with_new:
            if i != j and x[i, j].solution_value >= 0.5:
                j.arrival_time = t[j].solution_value

    obj_val = model.objective_value
    index_to_node = {node.index: node for node in nodes}  # nodes without the new customer
    old_obj = compute_route_cost(instance, km, index_to_node)

    return obj_val-old_obj, path, new_customer, nodes_with_new


def solve_TSP(instance,tech,nodes,new_cust,vehicles,times,km):
    tech=nodes[tech]
    nodes = nodes+[new_cust]

    # Solve the Model
    mod = build_TSP_problem(instance,nodes, times,km,tech,vehicles)
    sol = mod.solve(log_output=False, time_limit=20)

    if sol is None:
        return 999999,None, nodes[-1], nodes[:-1]

    sol_dict = json.loads(sol.export_as_json_string()) if sol else {}

    # Extract Routes
    vehicle_routes=[]

    for var in sol_dict['CPLEXSolution'].get('variables', []):
        if var['name'].startswith('route') and float(var['value']) > 0.5:
            parts = var['name'].split('_')
            i = int(parts[1].split('<')[1].split('>')[0])  # From node index
            j = int(parts[2].split('<')[1].split('>')[0])  # To node index
            vehicle_routes.append((i, j))
    
    obj_val=sol.objective_value
    index_to_node = {node.index: node for node in nodes}
   
    old_obj=compute_route_cost(instance,km,index_to_node)
        
        
    ordered_vehicle_paths = reconstruct_path(vehicle_routes,tech)

    # Update arrival times only in the copied nodes
    for node in nodes:
        for j in nodes:
            if node!=j:
                if mod.x_var[node, j].solution_value >= 0.5:
                    j.arrival_time = mod.t_var[j].solution_value

    return  obj_val-old_obj,ordered_vehicle_paths,nodes[-1],nodes[:-1]

