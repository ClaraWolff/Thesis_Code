import pickle
import folium
from folium.plugins import MarkerCluster

# Load penalty and schedule data
with open('Saves/saved_scheduler_data_penalties.pkl', 'rb') as f:
    data = pickle.load(f)

penalties = data['penalties']
nodes_days = data['nodes_days']
vehicle_days = data['vehicle_days']
days = data['days']

# --- Choose a day and find extremes ---
target_day = days[0]  # e.g., third day (adjust if needed)
penalty_dict = penalties[target_day]

# Sort customers by penalty
sorted_customers = sorted(penalty_dict.items(), key=lambda x: x[1])
low_id = sorted_customers[0][0]
high_id = sorted_customers[-1][0]
print(f"Low penalty (ID {low_id}): {penalty_dict[low_id]}")
print(f"High penalty (ID {high_id}): {penalty_dict[high_id]}")


# Get nodes
low_node = next(n for n in nodes_days[target_day] if n.global_index == low_id)
high_node = next(n for n in nodes_days[target_day] if n.global_index == high_id)

# Future days for insertion visualization
day_index = days.index(target_day)
future_days = days[day_index+1 : day_index+4]

# --- Set up folium map ---
m = folium.Map(location=[low_node.lat, low_node.long], zoom_start=8)

# Add large markers for selected customers
folium.Marker([low_node.lat, low_node.long], popup="Low Penalty", icon=folium.Icon(color='green', icon='ok')).add_to(m)
folium.Marker([high_node.lat, high_node.long], popup="High Penalty", icon=folium.Icon(color='red', icon='remove')).add_to(m)

# Plot other customers and technicians for next 3 days
for day in future_days:
    for node in nodes_days[day]:
        size = 3
        color = 'blue'
        if node.index in vehicle_days[day]:
            color = 'orange'  # Technician
        folium.CircleMarker(
            location=[node.lat, node.long],
            radius=size,
            color=color,
            fill=True,
            fill_opacity=0.5,
            popup=f"{day}, {'Tech' if node.index in vehicle_days[day] else 'Cust'}"
        ).add_to(m)

# Save map
m.save("Plots/penalty_example_map.html")
