import pandas as pd
import numpy as np
from folium import Marker, PolyLine
from docplex.mp.model import Model
import openrouteservice
from docplex.mp.conflict_refiner import ConflictRefiner
from TSP_heuristic import get_travel_time_matrix_global

class Node:
    def __init__(self, index, serv_time, time_window, lat, long,arrival=99999,ftr_by_type=None,work_order_type=None):
        self.index = index
        self.serv_time = serv_time
        self.time_window = time_window
        self.lat = lat
        self.long = long
        self.arrival_time=arrival
        self.ftr_by_type = ftr_by_type
        self.type=work_order_type
    
    def __repr__(self):
        return f"Node <{self.index}>"
    
# Setup ORS Client
API_KEY = 0000 #Insert API key here
client = openrouteservice.Client(key=API_KEY)

#Function to get correct customers and global distance+time matrices
def gather_nodes_and_matrix_with_global_index(dates, num_techs, num_cust_per_day, num_cust_tail_days):
    all_nodes = []
    day_to_node_ids = {}
    day_to_vehicles = {}
    leftover_customers={d:[] for d in dates}
    global_index = 0

    for date in dates:
        cust_count = num_cust_tail_days if date in dates[-6:] else num_cust_per_day

        nodes, vehicles,org_nodes = get_initial_data(date, num_techs, cust_count)

        for node in nodes:
            node.global_index = global_index
            global_index += 1

        leftover_customers[date]=[node for node in org_nodes if node not in nodes]
        for node in leftover_customers[date]:
            node.global_index = global_index
            global_index += 1


        all_nodes.extend(nodes)
        all_nodes.extend(leftover_customers[date])
        day_to_node_ids[date] = [node.global_index for node in nodes]
        day_to_vehicles[date]=vehicles

        

    times_global, km_global = get_travel_time_matrix_global(all_nodes)
    return all_nodes, day_to_node_ids, day_to_vehicles, km_global, times_global,leftover_customers

#Travel matrix function to get full matrix without timeout of API - function written by ChatGPT
def get_travel_time_matrix(nodes, speed_kmh=60, max_elements=3000):

    coordinates = [(node.long, node.lat) for node in nodes]
    n = len(coordinates)

    # Prepare empty full matrix
    full_distances = np.zeros((n, n))
    full_times = np.zeros((n, n))

    # Compute chunk size based on limit
    chunk_size = max(1, int(np.floor(np.sqrt(max_elements))))

    for i_start in range(0, n, chunk_size):
        for j_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            j_end = min(j_start + chunk_size, n)

            try:
                matrix = client.distance_matrix(
                    locations=coordinates,
                    sources=list(range(i_start, i_end)),
                    destinations=list(range(j_start, j_end)),
                    profile='driving-car',
                    metrics=['distance'],
                    units='m'
                )

                sub_dist = matrix['distances']
                for i_local, i_global in enumerate(range(i_start, i_end)):
                    for j_local, j_global in enumerate(range(j_start, j_end)):
                        dist_m = sub_dist[i_local][j_local]
                        dist_km = dist_m / 1000
                        time_hr = dist_km / speed_kmh
                        full_distances[i_global][j_global] = dist_km
                        full_times[i_global][j_global] = time_hr

            except Exception as e:
                print(f"Matrix chunk {i_start}:{i_end}, {j_start}:{j_end} failed -> {e}")

    return full_times, full_distances

def get_initial_data(chosen_date,num_techs,num_cust_init):
    # Load Data
    task_df = pd.read_excel("Data/CombinedTaskData.xlsx")
    techs_FTR=pd.read_excel("Data/FTR_per_resource_added.xlsx")
   
    ftr_dict = (techs_FTR
                .groupby('Resource')
                .apply(lambda df: dict(zip(df['Work Order Type'], df['FTR'])))
                .to_dict())

    # Filter Date
    my_date = pd.to_datetime(chosen_date).date()
    task_df_date= task_df[task_df['Start Time'].dt.date == my_date]

    # Select technicians
    selected_resources = task_df_date["Resource"].unique()[:num_techs]  
    task_df_techs = task_df_date[task_df_date["Resource"].isin(selected_resources)]

    # Define lat-long coordinates
    technician_coords = task_df_techs[["home_lat", "home_long"]].drop_duplicates().values.tolist()

    unique_techs = np.unique(task_df_techs["Resource"].tolist())

    #Keep only unique customer lat/long combinations
    task_df = task_df_techs.drop_duplicates(subset=["Latitude", "Longitude"])
    task_df = task_df.reset_index(drop=True)

    customer_coords = task_df[["Latitude", "Longitude"]].values.tolist()

    coordinates = {"tech" + str(i): technician_coords[i] for i in range(len(technician_coords))}
    coordinates.update({"cust" + str(i): customer_coords[i] for i in range(len(customer_coords))})

    technicians = [Node(i, 0, [7.5, 16], task_df_techs[task_df_techs["Resource"] == unique_techs[i]].iloc[0]["home_lat"],
                        task_df_techs[task_df_techs["Resource"] == unique_techs[i]].iloc[0]["home_long"],ftr_by_type=ftr_dict[unique_techs[i]]) for i in range(len(unique_techs))]

    start=task_df["Time Window Start"].dt.hour+task_df["Time Window Start"].dt.minute/60
    end=task_df["Time Window End"].dt.hour+task_df["Time Window End"].dt.minute/60

    start=start.reset_index(drop=True)
    end=end.reset_index(drop=True)

    normtime=task_df["Total Estimated Duration"]/60 #Minutes to hours
    normtime=normtime.reset_index(drop=True)

    customers = [Node(len(technicians) + i, normtime[i], [7.5, 16], task_df.iloc[i]["Latitude"], task_df.iloc[i]["Longitude"],work_order_type=task_df.iloc[i]["Work Order Type"]) for i in range(len(customer_coords))] #[start[i], end[i]]
    vehicles = list(range(len(technicians)))

    num_cust=num_cust_init
    num_cust_total=num_cust+30
    org_nodes = technicians + customers[:num_cust_total]


    times,km = get_travel_time_matrix(org_nodes) 

    illegal_count=0
    max_km = 80
    num_techs = len(technicians)
    num_total = len(technicians) + len(customers[:num_cust_total])

    # Build filtered list of customers who are within 80 km of at least 3 other nodes
    valid_customer_indices = []
    for i in range(num_techs, num_total):
        neighbor_count = sum(
        1 for j in range(num_total)
        if j != i and km[i][j] <= max_km
    )
        if neighbor_count >= 3:
            valid_customer_indices.append(i)
        else:
            print("removed: ",i)
            org_nodes[i].index=400+illegal_count
            illegal_count+=1

    # Rebuild customers and nodes with filtered customers
    filtered_customers = [org_nodes[i] for i in valid_customer_indices]

    for idx, node in enumerate(filtered_customers):
        node.index = num_techs+idx


    nodes = technicians + filtered_customers[:num_cust]

    task_df_techs=task_df_techs.iloc[np.random.permutation(len(task_df_techs))] 
    task_df = task_df_techs[:num_cust]

    return nodes,vehicles,org_nodes


# Define time windows and technician working hours
def build_VRPTW_problem_with_FTR(nodes,times,vehicles, km, penalties,lamb):
    e_i = {node.index: node.time_window[0] for node in nodes}
    l_i = {node.index: node.time_window[1] for node in nodes}
    H_start = {nodes[v].index: 7.5 for v in vehicles}  # Assume work starts at 7:30 AM
    H_end = {nodes[v].index: 16 for v in vehicles}  # Assume work ends at 16:00 PM


    customers = [n for n in nodes if n.index not in vehicles]
    depots = vehicles  # Each technician is a depot
    bigM = 25

    m = Model(name='MSVRPTW')
    m.parameters.mip.tolerances.absmipgap = 0
    m.parameters.mip.tolerances.mipgap = 0

    # Decision Variables
    x = m.binary_var_matrix(nodes, nodes, name='x')  # Routing variables
    y = m.binary_var_matrix(nodes, depots, name='y')  # Assignment variables
    t = m.continuous_var_dict(nodes, name='arrival_time')  # Arrival times

    m.x_var = x
    m.y_var = y
    m.t_var = t

    m.minimize(
    m.sum(km[i.global_index][j.global_index] * m.x_var[i, j] for i in nodes for j in nodes) +
    lamb * m.sum((1 - nodes[h].ftr_by_type.get(i.type, 1)) * y[i, h] * penalties[i.global_index]
              for i in customers for h in vehicles)
)


    # (25) Each trip has exactly one incoming arc
    for j in customers:
        m.add_constraint(m.sum(x[i, j] for i in nodes if i != j) == 1)

    # (26) Each trip has exactly one outgoing arc
    for i in customers:
        m.add_constraint(m.sum(x[i, j] for j in nodes if j != i) == 1)

    # (27) Each depot can start at most one route
    for h in depots:
        m.add_constraint(m.sum(x[nodes[h], j] for j in customers) <= 1)

    # (28) Link depot to assigned customer
    for h in depots:
        for j in customers:
            m.add_constraint(x[nodes[h], j] <= y[j, h])

    # (29) Link customer to depot
    for h in depots:
        for i in customers:
            m.add_constraint(x[i, nodes[h]] <= y[i, h])

    # (30) If i->j exists, they must share depot
    for h in depots:
        for i in customers:
            for j in customers:
                if i != j:
                    m.add_constraint(y[i, h] + x[i, j] - y[j, h] <= 1)

    # (31) Symmetric to (30)
    for h in depots:
        for i in customers:
            for j in customers:
                if i != j:
                    m.add_constraint(y[j, h] + x[i, j] - y[i, h] <= 1)

    # (32) Each trip is assigned to exactly one depot
    for i in customers:
        m.add_constraint(m.sum(y[i, h] for h in depots) == 1)

    # -----------------------
    # Time Constraints
    # -----------------------

    # Time feasibility: if route i->j exists, t_j >= t_i + service + travel
    for i in customers:
        for j in nodes:
            if i != j:
                m.add_constraint(
                    t[j] >= t[i] + i.serv_time + times[i.global_index][j.global_index] - bigM * (1 - x[i, j])
                )

    # Time windows at customers
    for i in customers:
        m.add_constraint(e_i[i.index] <= t[i])
        m.add_constraint(t[i] <= l_i[i.index])

    # Depot return time must match technician's end time
    for v in depots:
        m.add_constraint(t[nodes[v]] <= H_end[nodes[v].index])


    # First customer must be reachable from depot start time
    for v in depots:
        for j in customers:
            m.add_constraint(
                t[j] >= H_start[nodes[v].index] + times[nodes[v].global_index][j.global_index] - bigM * (1 - x[nodes[v], j])
            )

    for i in nodes:
        for j in customers:
            if i != j:
                m.add_constraint(
                    km[i.global_index, j.global_index] * x[i, j] <= 80
                )

    return m


def build_VRPTW_problem(nodes,times,vehicles, km):
    e_i = {node.index: node.time_window[0] for node in nodes}
    l_i = {node.index: node.time_window[1] for node in nodes}
    H_start = {v: 7.5 for v in vehicles}  # Assume work starts at 7:30 AM
    H_end = {v: 16 for v in vehicles}  # Assume work ends at 16:00 PM


    customers = [n for n in nodes if n.index not in vehicles]
    depots = vehicles  # Each technician is a depot
    bigM = 1000

    m = Model(name='MSVRPTW')
    m.parameters.mip.tolerances.absmipgap = 0
    m.parameters.mip.tolerances.mipgap = 0

    # Decision Variables
    x = m.binary_var_matrix(nodes, nodes, name='x')  # Routing variables
    y = m.binary_var_matrix(customers, depots, name='y')  # Assignment variables
    t = m.continuous_var_dict(nodes, name='arrival_time')  # Arrival times

    m.x_var = x
    m.y_var = y
    m.t_var = t

    # Objective: minimize total travel distance
    m.minimize(m.sum(km[i.global_index][j.global_index] * m.x_var[i, j] for i in nodes for j in nodes))

    # (25) Each trip has exactly one incoming arc
    for j in customers:
        m.add_constraint(m.sum(x[i, j] for i in nodes if i != j) == 1)

    # (26) Each trip has exactly one outgoing arc
    for i in customers:
        m.add_constraint(m.sum(x[i, j] for j in nodes if j != i) == 1)

    # (27) Each depot can start at most one route
    for h in depots:
        m.add_constraint(m.sum(x[nodes[h], j] for j in customers) <= 1)

    # (28) Link depot to assigned customer
    for h in depots:
        for j in customers:
            m.add_constraint(x[nodes[h], j] <= y[j, h])

    # (29) Link customer to depot
    for h in depots:
        for i in customers:
            m.add_constraint(x[i, nodes[h]] <= y[i, h])

    # (30) If i->j exists, they must share depot
    for h in depots:
        for i in customers:
            for j in customers:
                if i != j:
                    m.add_constraint(y[i, h] + x[i, j] - y[j, h] <= 1)

    # (31) Symmetric to (30)
    for h in depots:
        for i in customers:
            for j in customers:
                if i != j:
                    m.add_constraint(y[j, h] + x[i, j] - y[i, h] <= 1)

    # (32) Each trip is assigned to exactly one depot
    for i in customers:
        m.add_constraint(m.sum(y[i, h] for h in depots) == 1)

    # -----------------------
    # Time Constraints
    # -----------------------

    # Time feasibility: if route i->j exists, t_j >= t_i + service + travel
    for i in customers:
        for j in nodes:
            if i != j:
                m.add_constraint(
                    t[j] >= t[i] + i.serv_time + times[i.global_index][j.global_index] - bigM * (1 - x[i, j])
                )

    # Time windows at customers
    for i in customers:
        m.add_constraint(e_i[i.index] <= t[i])
        m.add_constraint(t[i] <= l_i[i.index])

    # Depot return time must match technician's end time
    for v in depots:
        m.add_constraint(t[nodes[v]] <= H_end[nodes[v].index])

    # First customer must be reachable from depot start time
    for v in depots:
        for j in customers:
            m.add_constraint(
                t[j] >= H_start[nodes[v].index] + times[nodes[v].global_index][j.global_index] - bigM * (1 - x[nodes[v], j])
            )

    # Optional: max travel distance (e.g., 80 km)
    for i in nodes:
        for j in customers:
            if i != j:
                m.add_constraint(
                    km[i.global_index, j.global_index] * x[i, j] <= 80
                )


    return m


def extract_vehicle_routes_and_paths(nodes, x_var, vehicles):
    """Returns:
    - vehicle_routes: raw arc list per vehicle
    - ordered_vehicle_paths: full path in order per vehicle
    """
    
    arcs = {(i.index, j.index) for i in nodes for j in nodes
            if i != j and x_var[i, j].solution_value > 0.5}

    vehicle_routes = {v: [] for v in vehicles}
    ordered_paths = {v: [] for v in vehicles}

    for v in vehicles:
        start = v  # depot index
        path = [start]
        current = start
        visited = set([start])

        while True:
            next_nodes = [j for (i, j) in arcs if i == current and (j not in visited or j==start)]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            vehicle_routes[v].append((current, next_node))
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        ordered_paths[v] = path

    return vehicle_routes, ordered_paths


# Solve the Model
def get_initial_sol(all_nodes, node_ids_for_day, km_global, times_global, vehicles):
    nodes = [n for n in all_nodes if n.global_index in node_ids_for_day]

    mod = build_VRPTW_problem(nodes, times_global, vehicles, km_global)
    sol = mod.solve(log_output=True)
    if sol is None:
        cr = ConflictRefiner()
        conflicts = cr.refine_conflict(mod)
        print(conflicts.display())
        print("Model is infeasible")

    vehicle_routes, ordered_vehicle_paths = extract_vehicle_routes_and_paths(nodes, mod.x_var, vehicles)

    return nodes, vehicles, ordered_vehicle_paths, km_global, times_global

def get_FTR_sol(nodes,vehicles,km,times,penalties,lamb):     
    mod = build_VRPTW_problem_with_FTR(nodes, times, vehicles,km,penalties,lamb)
    sol = mod.solve(log_output=True)
    if sol is None:
        cr=ConflictRefiner()
        conflicts=cr.refine_conflict(mod)
        print(conflicts.display())
        print("Model is infeasible")
        
    print(f"Objective from model: {sol.objective_value}")

    for node in nodes:
        if node.index not in vehicles:
            node.arrival_time = mod.t_var[node].solution_value


    vehicle_routes, ordered_vehicle_paths = extract_vehicle_routes_and_paths(nodes, mod.x_var, vehicles)
    
    return nodes,vehicles,ordered_vehicle_paths
