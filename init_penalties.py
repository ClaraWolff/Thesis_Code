import pickle
import copy
import TSP_heuristic


# Load init
with open('Saves/saved_scheduler_data.pkl', 'rb') as f:
    data = pickle.load(f)

days = data['days']
days_init = data['days_init']
customer_days = data['customer_days']
nodes_days = data['nodes_days']
vehicle_days = data['vehicle_days']
leftover_customers = data['leftover_customers']

with open("Saves/global_distance_matrices.pkl", "rb") as f:
    data = pickle.load(f)
    times_global = data['times']
    km_global = data['km']


print("Calculating penalties")
penalties = {
    d: {n.global_index: 0 for n in nodes_days[d] if n.index not in vehicle_days[d]}
    for d in days[:len(days) - 3]
}
for d in range(len(days)-3):
    day=days[d]
    print("penalities for day: ", day, "for days: ",days[d+1:d+4])
    for c in customer_days[day]:
        penalties_day=[]
        for nd in days[d+1:d+4]:
            new_node=copy.deepcopy(c)
            new_node.index=max([node.index for node in nodes_days[nd]])+1
            val,route,min_route,nodes_copy=TSP_heuristic.get_obj_dif_for_new_customer(new_node,days_init[nd],nodes_days[nd],vehicle_days[nd],times=times_global,km=km_global)
            penalties_day.append(val)
        penalties[day][c.global_index]=max(0.5,min(penalties_day)) #So we don't include negative penalties. 


print(penalties)


# Save all necessary variables
with open('Saves/saved_scheduler_data_penalties.pkl', 'wb') as f:
    pickle.dump({
        'days': days,
        'days_init': days_init,
        'customer_days': customer_days,
        'nodes_days': nodes_days,
        'vehicle_days': vehicle_days,
        'leftover_customers': leftover_customers,
        'penalties': penalties,
        'km_global': km_global,
        'times_global': times_global
    }, f)

print("Saved initialization results to 'saved_scheduler_data_penalties.pkl'")