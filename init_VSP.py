import pickle
import two_index_VSP
from two_index_VSP import gather_nodes_and_matrix_with_global_index
from TSP import compute_route_cost

# - Assume all tasks have to be solved within 3 days
#Initialize the different days
days=['2025-01-13','2025-01-14','2025-01-15','2025-01-16','2025-01-17','2025-01-20','2025-01-21','2025-01-22','2025-01-24','2025-01-27','2025-01-28'] 

days_init={d:[] for d in days}
customer_days={d:[] for d in days}
nodes_days={d:[] for d in days}
vehicle_days={d:[] for d in days}
km_days={d:[] for d in days}
times_days={d:[] for d in days}


#Get subset customers and global distance matrices for all nodes
all_nodes, day_to_node_ids, day_to_vehicles, km_global, times_global,leftover_customers = gather_nodes_and_matrix_with_global_index(
    dates=days,
    num_techs=25, #Just an upper bound, actually have 8-10
    num_cust_per_day=19,
    num_cust_tail_days=15
)

#Solve mdvsp for each day with small subset of customers to have baseline for each day
for d in days:
    print("solve initial for day: ", d)

    vehicles = day_to_vehicles[d]
    sol=two_index_VSP.get_initial_sol(
        all_nodes=all_nodes,
        node_ids_for_day=day_to_node_ids[d],
        km_global=km_global,
        times_global=times_global,
        vehicles=vehicles
        ) 
    
    nodes,vehicles=sol[0],sol[1]
    nodes_days[d]=nodes
    vehicle_days[d]=vehicles
    customers = [n for n in nodes if n.index not in vehicles]
    customer_days[d]=customers

    days_init[d]=sol[2]
    km_days[d]=sol[3]
    times_days[d]=sol[4]
    

#compute total distance
total_km = 0
for day in days[:len(days)-6]:
    print(day)
    for route in days_init[day].values():
        index_to_node_baseline = {node.index: node for node in nodes_days[day]}
        next_cost=compute_route_cost(route,km_global,index_to_node_baseline)
        total_km += next_cost

print("Total cost:",total_km)

#Save for later use
with open("Saves/global_distance_matrices.pkl", "wb") as f:
    pickle.dump({
        'times': times_global,
        'km': km_global,
        'nodes': all_nodes
    }, f)

with open('Saves/saved_scheduler_data.pkl', 'wb') as f:
    pickle.dump({
        'days': days,
        'days_init': days_init,
        'customer_days': customer_days,
        'nodes_days': nodes_days,
        'vehicle_days': vehicle_days,
        'km_days': km_days,
        'times_days': times_days,
        'leftover_customers': leftover_customers
    }, f)
