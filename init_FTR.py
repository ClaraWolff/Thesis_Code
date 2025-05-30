import pickle
import copy
import TSP_heuristic
import time
import two_index_VSP
from TSP import compute_route_cost
from Scheduling_days import compute_ftr_weighted_penalty
from Scheduling_days import compute_expected_revisits

lamb=60

def assign_time_window(arrival):
    if 7.5 <= arrival < 12.0:
        return [7.5, 12.5]  # Morning
    elif 12.0 <= arrival < 16.0:
        return [12.0, 16.0]  # Afternoon
    elif 16.0 <= arrival < 18.0:
        return [16.0, 18.0]  # Evening
    else:
        return [7.5, 16.0]  # Default to all day if outside known slots

#Load initialization and penalties
with open('Saves/saved_scheduler_data_penalties.pkl', 'rb') as f:
    data = pickle.load(f)

days = data['days']
days_init = data['days_init']
days_init_baseline = copy.deepcopy(days_init)
customer_days = data['customer_days']
nodes_days = data['nodes_days']
nodes_days_baseline=copy.deepcopy(nodes_days)
vehicle_days = data['vehicle_days']
leftover_customers = data['leftover_customers']
penalties=data['penalties']
km_global=data['km_global']
times_global=data['times_global']

total_revisits_baseline = sum(
            compute_expected_revisits(nodes_days_baseline[day],days_init_baseline[day])
            for day in days[:len(days)-3]
        )

weighted_penalty_before = compute_ftr_weighted_penalty(
    {day: nodes_days_baseline[day] for day in days[:len(days) - 3]},
    {day: days_init_baseline[day] for day in days[:len(days) - 3]},
    {day: penalties[day] for day in days[:len(days) - 3]}
)

print("Before penalties",days_init)

total_km = 0
for day in days[:len(days)-3]:
    for route in days_init_baseline[day].values():
        index_to_node_baseline = {node.index: node for node in nodes_days_baseline[day]}
        total_km += compute_route_cost(route,km_global,index_to_node_baseline)
print("Total objective before: ", total_km)


#Solve again with penalties
for d in days[:len(days)-3]:
    print("solving day: ",d)
    nodes, vehicles, ordered_vehicle_paths=two_index_VSP.get_FTR_sol(nodes_days[d],vehicle_days[d],km_global,times_global,penalties[d],lamb=lamb) 
    nodes_days[d]=nodes
    days_init[d]=ordered_vehicle_paths
    for node in nodes_days[d]:
        if node.index not in vehicle_days[d]:  # Skip technicians for assigning time windows
            node.time_window = assign_time_window(node.arrival_time)


total_km = 0
for day in days[:len(days)-3]:
    for route in days_init_baseline[day].values():
        index_to_node_baseline = {node.index: node for node in nodes_days_baseline[day]}
        next_cost=compute_route_cost(route,km_global,index_to_node_baseline)
        total_km += next_cost


print("Total distance objective baseline: ", total_km)

total_km = 0
for day in days[:len(days)-3]:
    for route in days_init[day].values():
        index_to_node = {node.index: node for node in nodes_days[day]}
        next_cost_new=compute_route_cost(route, km_global,index_to_node)
        total_km += next_cost_new

weighted_penalty_after = compute_ftr_weighted_penalty(
    {day: nodes_days[day] for day in days[:len(days) - 3]},
    {day: days_init[day] for day in days[:len(days) - 3]},
    {day: penalties[day] for day in days[:len(days) - 3]}
)
total_revisits_after = sum(
            compute_expected_revisits(nodes_days[day],days_init[day])
            for day in days[:len(days)-3]
        )


path=f'Saves/lambda{lamb}_saved_scheduler_data_FTR.pkl'
# Save all necessary variables
added_km = total_km - (
    sum(
        compute_route_cost(route, km_global, {node.index: node for node in nodes_days_baseline[day]})
        for day in days[:len(days)-3]
        for route in days_init_baseline[day].values()
    )
)

results = {
    'days': days,
    'days_init': days_init,
    'days_init_baseline': days_init_baseline,
    'customer_days': customer_days,
    'nodes_days': nodes_days,
    'vehicle_days': vehicle_days,
    'leftover_customers': leftover_customers,
    'penalties': penalties,
    
    # Model parameters
    'ftr_penalty_lambda': lamb,
    
    # Objectives (raw)
    'total_km_before': total_km - added_km,
    'total_km_after': total_km,
    'added_km': added_km,
    
    # Revisit estimates
    'expected_revisits_before': total_revisits_baseline,
    'expected_revisits_after': total_revisits_after,
    'expected_revisits_reduction': total_revisits_baseline - total_revisits_after,
    
    # FTR penalty terms
    'ftr_weighted_penalty_before': weighted_penalty_before,
    'ftr_weighted_penalty_after': weighted_penalty_after,
    'ftr_weighted_penalty_reduction': weighted_penalty_before - weighted_penalty_after,
    
    # Meta
    'n_customers_ftr': sum(
        1 for day in days for node in nodes_days[day]
        if node.ftr_by_type is not None
    )
}


with open(path, 'wb') as f:
    pickle.dump(results, f)

print("Saved initialization results to 'lambda?_saved_scheduler_data_FTR.pkl'")
