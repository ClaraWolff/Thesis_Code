
import copy
import TSP_heuristic
import time
import two_index_VSP
import numpy as np
import pickle
import folium
from folium import Marker, PolyLine
import openrouteservice
from TSP import compute_route_cost


def assign_time_window(arrival):
    if 7.5 <= arrival < 12.0:
        return [7.5, 12.5]  # Morning
    elif 12.0 <= arrival < 16.0:
        return [12.0, 16.0]  # Afternoon
    elif 16.0 <= arrival < 18.0:
        return [16.0, 18.0]  # Evening
    else:
        return [7.5, 16.0]  # Fallback


def compute_ftr_weighted_penalty(nodes_dict, routes_dict, penalties_dict):
    total_penalty = 0
    for d, tech_routes in routes_dict.items():
        node_lookup = {n.index: n for n in nodes_dict[d]}
        for tech_id, route in tech_routes.items():
            tech_node = node_lookup.get(tech_id)
            if tech_node is None or not hasattr(tech_node, "ftr_by_type"):
                continue

            for node_idx in route:
                if node_idx == tech_id:
                    continue  # skip depot

                cust = node_lookup.get(node_idx)
                if cust is None or not hasattr(cust, "type") or not hasattr(cust, "global_index"):
                    continue
    
                ftr_score = tech_node.ftr_by_type.get(cust.type, 1)
                penalty_pi = penalties_dict[d].get(cust.global_index, 0)

                total_penalty += (1 - ftr_score) * penalty_pi

    return total_penalty

def compute_expected_revisits(nodes, day_routes):
    index_to_node = {n.index: n for n in nodes}
    revisits = 0

    for tech_id, route in day_routes.items():
        technician = index_to_node.get(tech_id)  # Get the technician node by vehicle ID
        #print(technician)
        if technician is None or not hasattr(technician, 'ftr_by_type'):
            print(f"Warning: Technician {tech_id} not found or missing ftr_by_type")
            continue

        for node_idx in route:
            if node_idx == tech_id:
                continue  # Skip depot/technician node itself

            customer = index_to_node.get(node_idx)
    
            if customer is None or not hasattr(customer, 'type'):
                print(f"Warning: Customer {node_idx} not found or missing type")
                continue

            task_type = customer.type
            ftr_score = technician.ftr_by_type.get(task_type, 1) # Assume 1 if no ftr for that task type
    
            revisits += (1 - ftr_score)

    return revisits


def get_best_baseline_route(customer, current_day_index, days, days_init, nodes_days, vehicle_days, times, km):
    global next_baseline_global_index

    best_val = float("inf")
    best_route = []
    best_day = None
    best_tech_id = None
    best_nodes_copy = []

    future_days = days[current_day_index: current_day_index +3]
    for d in future_days:
        print("baseline day:",d)
        new_node = copy.deepcopy(customer)
        new_node.index = max(n.index for n in nodes_days[d]) + 1
        new_node.global_index=customer.global_index

        try:
            val, route, tech_id, nodes_copy = TSP_heuristic.get_obj_dif_for_new_customer(
                new_node,
                days_init[d],
                nodes_days[d],
                vehicle_days[d],
                times,
                km
            )

            if val < best_val:
                best_val = val
                best_route = route
                best_day = d
                best_tech_id = tech_id
                best_nodes_copy = copy.deepcopy(nodes_copy)
                best_new_node = new_node

        except Exception as e:
            print(f"Baseline route failed on {day}: {e}")
            continue

    if best_val==float('inf'):
        return best_val,None,None,None,None,None
    
    best_new_node.time_window = assign_time_window(new_node.arrival_time)
    
    return best_val, best_route, best_tech_id, best_day, best_new_node, best_nodes_copy

def calculate_penalty_for_customer(customer, current_day_index, days, days_init, nodes_days, vehicle_days):

    penalties_day = []
    future_days = days[current_day_index: current_day_index+3]

    for future_day in future_days:
        new_node = copy.deepcopy(customer)
        new_node.index = max(n.index for n in nodes_days[future_day]) + 1
        new_node.global_index=customer.global_index

        try:
            val,_, _, _ = TSP_heuristic.get_obj_dif_for_new_customer(
                new_node,
                days_init[future_day],
                nodes_days[future_day],
                vehicle_days[future_day],
                times_global,
                km_global
            )
            penalties_day.append(val)
        except Exception as e:
            print(f"Penalty calculation failed for {future_day}: {e}")
            penalties_day.append(float("inf"))

    min_penalty = max(0.5, min(penalties_day)) if penalties_day else float("inf")
    return min_penalty

def get_real_route(start, end):
    """Queries OpenRouteService for real driving route"""
    API_KEY = 0000 #Input API key here
    client = openrouteservice.Client(key=API_KEY)
   
    try:
        time.sleep(1)
        route = client.directions(
            coordinates=[(start.long, start.lat), (end.long, end.lat)],
            profile="driving-car",
            format="geojson"
        )
        return route['features'][0]['geometry']['coordinates']  # List of (long, lat) points

    except:

        return []

def plot_real_routes(map_obj, vehicle_routes, nodes):
    route_colors = ["blue", "red", "green", "purple", "magenta", "brown", "black"]
    for vehicle, path in vehicle_routes.items():
        vehicle_color = route_colors[vehicle % len(route_colors)]
        for i, j in zip(path[:-1], path[1:]):   
            start = nodes[i]
            end = nodes[j]
            real_route = get_real_route(start, end)
            if real_route:
                real_route = [(lat, lon) for lon, lat in real_route]  # Convert (lon, lat) -> (lat, lon)
                PolyLine(real_route, color=vehicle_color, weight=2.5, opacity=0.8).add_to(map_obj)

def plot_day(day_idx, day, nodes, vehicles, vehicle_routes):
    # Use first node to center the map
    if not nodes:
        print(f"No nodes for {day}, skipping.")
        return

    m = folium.Map(location=[nodes[0].lat, nodes[0].long], tiles='openstreetmap', zoom_start=10)
    
    for vehicle, path in vehicle_routes.items():
        for order, node_index in enumerate(path):
            node = nodes[node_index]
            if node.index not in vehicles:
                visit_number = order
                icon = folium.DivIcon(
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                    html=f'<div style="font-size: 14px; color: white; background: blue; '
                         f'border-radius: 50%; width: 25px; height: 25px; text-align: center; '
                         f'line-height: 25px; font-weight: bold;">{visit_number}</div>'
                )
            else:
                icon = folium.Icon(color="red", icon="home")

            arrival_time = node.arrival_time if node.index not in vehicles else "N/A"
            return_time = node.arrival_time if node.index in vehicles else "N/A"
            label = (
                f"Technician {node.index}<br>Departure: {node.time_window[0]}<br>Return: {return_time:.2f}"
                if node.index in vehicles else
                f"Customer {node.index}<br>Expected Arrival: {arrival_time:.2f} <br>"
                f"Window: {node.time_window[0]} - {node.time_window[1]}<br>Est. duration: {node.serv_time:.2f}"
            )
            Marker(location=[node.lat, node.long], icon=icon, popup=label).add_to(m)

    # draw real driving routes:
    plot_real_routes(m, vehicle_routes, nodes)

    filename = f"Plots/technician_routes_day_{day_idx+1}_80_50000.html"
    m.save(filename)
    print(f"Saved map for {day} as {filename}")

if __name__=="__main__":
    try:
        lamb = float(input("Enter lambda value: "))
    except ValueError:
        print("Lambda not entered")
    print("Lambda2:",lamb)

    # Load saved schedule and structure from FTR
    with open('Saves/lambda80_saved_scheduler_data_FTR.pkl', 'rb') as f:
        data = pickle.load(f)

    days = data['days']
    days_init = data['days_init']
    nodes_days = data['nodes_days']
    vehicle_days = data['vehicle_days']
    leftover_customers = data['leftover_customers']
    N = len(days) - 6

    # Load baseline for comparison
    with open('Saves/saved_scheduler_data_penalties.pkl', 'rb') as f:
        baseline_data = pickle.load(f)

    original_penalties = baseline_data['penalties']

    # Load global matrix
    with open("Saves/global_distance_matrices.pkl", "rb") as f:
        distance_data = pickle.load(f)
        times_global = distance_data['times']
        km_global = distance_data['km']

    penalties=original_penalties

    days_with_penalties = days[:N+3]

    penalty_ftr = compute_ftr_weighted_penalty(
        {d: nodes_days[d] for d in days_with_penalties},
        {d: days_init[d] for d in days_with_penalties},
        penalties
    )

    print("BEFORE INSERTION")
    print("penalty FTR: ",penalty_ftr)
    total_revisits_ftr_all_customers=0

    differences={}
    num=0
    for d in range(N):
        day=days[d]
        print("solve for day: ",day)
        
        for c in leftover_customers[day][:]: 
            num+=1
            day_vals={da:9999 for da in days[d:d+3]}
            day_route={da:{v:[] for v in days_init[da].keys()} for da in days[d:d+3]}
            min_routes={da:99999 for da in days[d:d+3]}
            new_nodes={da:None for da in days[d:d+3]}
            nodes_copies={da:[] for da in days[d:d+3]}

            penalty_for_c= calculate_penalty_for_customer(customer=c,
                current_day_index=d,
                days=days,
                days_init=days_init,
                nodes_days=nodes_days,
                vehicle_days=vehicle_days
            )   

            print("penalty for c", penalty_for_c)



            for solve_day in days[d:d+3]:
                print("solve_day: ",solve_day)
                new_node=copy.deepcopy(c)
                new_node.index=max([node.index for node in nodes_days[solve_day]])+1
                print(new_node.index)
            
                val, route, min_route, nodes_copy = TSP_heuristic.get_obj_dif_for_new_customer_FTR(
                new_node,
                days_init[solve_day],
                nodes_days[solve_day],
                vehicle_days[solve_day],
                {new_node.global_index: penalty_for_c},  # single-customer penalty
                times=times_global,
                km=km_global,
                lambda_ftr=lamb
                )
                
                day_vals[solve_day]=val
                day_route[solve_day][min_route]=route
                min_routes[solve_day]=min_route
                new_nodes[solve_day]=new_node
                nodes_copies[solve_day]=copy.deepcopy(nodes_copy) #Includes the new node
        
                
                
            if min(day_vals.values()) > 9999:
                print(f"Skipping customer {c.global_index} due to infeasibility")
                continue
            
                    
            min_day=min(day_vals,key=day_vals.get)
            days_init[min_day][min_routes[min_day]]=day_route[min_day][min_routes[min_day]]
            inserted_customer = new_nodes[min_day]
            penalties[min_day][new_node.global_index] = penalty_for_c

            inserted_customer.time_window = assign_time_window(inserted_customer.arrival_time)
            for node in nodes_copies[min_day]:
                if node.global_index == inserted_customer.global_index:
                    node.time_window = inserted_customer.time_window
                    break
            nodes_days[min_day]=nodes_copies[min_day] #Need to overwrite to include new arrival times
    
            print("Assigned route:", day_route[min_day][min_routes[min_day]], min_day)
            # Insert c into FTR version
            days_init[min_day][min_routes[min_day]] = day_route[min_day][min_routes[min_day]]
            #Compute the penalty part of the objective
            days_with_penalties = days[:len(days)-3]
            days_with_penalties = days[:len(days)-3]
            penalty_ftr = compute_ftr_weighted_penalty(
                {da: nodes_days[da] for da in days_with_penalties},
                {da: days_init[da] for da in days_with_penalties},
                penalties
            )

            print("penalty after: ",penalty_ftr)

            total_revisits_ftr = sum(
                compute_expected_revisits(nodes_days[da], days_init[da])
                for da in days[:N+3]
            )

            total_km_ftr = 0
            for da in days[:N+3]:
                for route in days_init[da].values():
                    index_to_node = {node.index: node for node in nodes_days[da]}
                    next_cost_new=compute_route_cost(route, km_global,index_to_node)
                    #print(route,next_cost_new)
                    total_km_ftr += next_cost_new

            print("revisits after:",total_revisits_ftr)
            print("total km ftr:", total_km_ftr)
            total_revisits_ftr_all_customers=total_revisits_ftr
            differences[num] = {
            "number node": c.global_index,
            "revisits after": total_revisits_ftr,
            "penalty after": penalty_ftr,
            "day": day
            }
            
    print("Total revisits ftr included:",total_revisits_ftr_all_customers)
    print(days_init)

    total_revisits_ftr = sum(
        compute_expected_revisits(nodes_days[d], days_init[d])
        for d in days[:N+3]
    )

    print("revisits after",total_revisits_ftr)

    total_km_ftr = 0
    for d in days[:N+3]:
        for route in days_init[d].values():
            index_to_node = {node.index: node for node in nodes_days[d]}
            next_cost_new=compute_route_cost(route, km_global,index_to_node)
            total_km_ftr += next_cost_new


    with open("Saves/lambda80_50000_penalties_TSP_FTR.pkl", "wb") as f:
        pickle.dump(penalties, f)

    print("lambda?_?_Saved penalties to penalties_FTR.pkl")

    with open("Saves/lambda80_50000_differences_revisits.pkl", "wb") as f:
        pickle.dump(differences, f)

    print("Saved individual differences to lambda?_?_differences_revisits.pkl")

    totals = {
    "ftr_penalty_lambda": lamb,  # the lambda value used
    "ftr_weighted_penalty_after": penalty_ftr,
    "total_km_after": total_km_ftr,
    "expected_revisits_after": total_revisits_ftr
}

    with open("Saves/lambda80_50000_total_revisit_summary.pkl", "wb") as f:
        pickle.dump(totals, f)

    print("Saved total revisit summary to lambda?_?_total_revisit_summary.pkl")


    print("Plotting")

    #Maps
    for idx, day in enumerate(days[:len(days)-6]): #Not for the last 6 days that are just there to calculate penalty and schedule
        nodes = nodes_days[day]
        vehicles = vehicle_days[day]
        vehicle_routes = days_init[day] 
        plot_day(idx, day, nodes, vehicles, vehicle_routes)
