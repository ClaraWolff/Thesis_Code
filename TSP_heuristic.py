import TSP 
import openrouteservice
import numpy as np
import time

# Setup ORS Client
API_KEY = 0000 #Insert API key here
client = openrouteservice.Client(key=API_KEY)



def get_travel_time_matrix_global(nodes):  #Function generated by ChatGPT:
    """
    Returns two matrices (times and distances) indexed by node.global_index.
    The output matrices are of size (max_global_index + 1) x (max_global_index + 1).
    """
    coordinates = [(node.long, node.lat) for node in nodes]
    global_indices = [node.global_index for node in nodes]

    max_global_index = max(global_indices)
    n = len(nodes)

    # Initialize full matrices
    full_times = np.full((max_global_index + 1, max_global_index + 1), np.inf)
    full_dists = np.full((max_global_index + 1, max_global_index + 1), np.inf)

    # Split into ORS-sized batches
    max_rows = 50  # ORS limit for free users
    max_cols = 50
    for i_start in range(0, n, max_rows):
        print("Wait for API cooldown")
        time.sleep(20)
        for j_start in range(0, n, max_cols):
            i_end = min(i_start + max_rows, n)
            j_end = min(j_start + max_cols, n)

            sources = list(range(i_start, i_end))
            destinations = list(range(j_start, j_end))

            try:
                result = client.distance_matrix(
                    locations=coordinates,
                    sources=sources,
                    destinations=destinations,
                    metrics=["distance", "duration"],
                    profile="driving-car",
                    units="km"
                )

                # Fill in the full matrix using global indices
                for i_local, i in enumerate(sources):
                    gi = nodes[i].global_index
                    for j_local, j in enumerate(destinations):
                        gj = nodes[j].global_index
                        dist = result["distances"][i_local][j_local]
                        dur = result["durations"][i_local][j_local]
                        full_dists[gi][gj] = dist
                        full_times[gi][gj] = dur / 3600  # convert sec -> hours

            except Exception as e:
                print(f"ORS matrix call failed for block ({i_start}-{i_end}, {j_start}-{j_end}): {e}")
                time.sleep(10)  # Cooldown before retry

    return full_times, full_dists


def add_new_customer(i,ordered_vehicle_paths,nodes,vehicles,times,km):
    tsps={v: [] for v in vehicles}
    obj_val_difs={v: [] for v in vehicles}
    node_copies={v:[] for v in vehicles}
    new_nodes_copies={v:[] for v in vehicles}
    new_cust_node=i
    
    for v in ordered_vehicle_paths.keys():
        route=ordered_vehicle_paths[v]
        obj,tsp_sol,new_customer_node_copy,nodes_copy=TSP.solve_TSP(route,v,nodes,new_cust_node,vehicles,times,km)
        tsps[v]=tsp_sol
        obj_val_difs[v]=obj
        node_copies[v]=nodes_copy
        new_nodes_copies[v]=new_customer_node_copy

    min_obj_dif=min(obj_val_difs.values())
    min_route=min(obj_val_difs, key=obj_val_difs.get)
    nodes=node_copies[min_route]
    
    return min_obj_dif,tsps[min_route],min_route,nodes
    
def get_obj_dif_for_new_customer(i,ordered_vehicle_paths,nodes,vehicles,times,km):
    #nodes_temp=nodes+[i]
    obj_dif,route,min_route,nodes=add_new_customer(i,ordered_vehicle_paths,nodes,vehicles,times,km)

    
    return obj_dif,route,min_route,nodes


def get_obj_dif_for_new_customer_FTR(i, ordered_vehicle_paths, nodes, vehicles, penalties,times,km,lambda_ftr):
    #nodes_temp = nodes + [i]

    tsps = {}
    obj_val_difs = {}
    node_copies = {}

    for v in ordered_vehicle_paths.keys():
        route = ordered_vehicle_paths[v]
        #print("route:",route)
        penalty = penalties.get(i.global_index, 0)
        obj, tsp_sol, new_customer_copy, nodes_copy = TSP.solve_TSP_with_FTR(
            route, v, nodes, i, vehicles, times, km, penalty, lambda_ftr
        )
        tsps[v] = tsp_sol
        obj_val_difs[v] = obj
        node_copies[v] = nodes_copy
    
    min_route = min(obj_val_difs, key=obj_val_difs.get)
    return obj_val_difs[min_route], tsps[min_route], min_route, node_copies[min_route]
