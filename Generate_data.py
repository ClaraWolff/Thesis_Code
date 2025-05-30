import pandas as pd
import Helper_funcs as funcs

def read_normtider():
    dfs = pd.read_excel('Data\Source\Opgave typer tid.xlsx',header=0)
    return dfs[["External Incident ID","Name",'Estimated Duration']]

def read_task_data():
    df = pd.read_excel('Data\Source\FilerV2\FSM_Active Work Orders_Datafil.xlsx',header=0)
    df=df[df["System Status"]=="Completed"]
    df=df[df["External Order ID"].notnull()]
    
    df=df[["External Order ID","Work Order Number","Created On","Work Order Type","Primary Incident Type","Street 1","Postal Code","City","Latitude","Longitude","Time Window Start","Time Window End","Total Estimated Duration","Primary Incident Estimated Duration"]]
    return df

def read_sample_home_adresses():
    # Load the original file
    df = pd.read_excel('Data\Source\FSM_Active Bookable Resource Bookings_Datafil.xlsx',header=0)
    df = df[df["Booking Status"] == "Udført"]
    df = df[["Resource", "Work Order", "Travel Distance in KM"]]

    # Only take sample from distances less than 30 km. away from previous customer
    df["Travel Distance in KM"] = pd.to_numeric(df["Travel Distance in KM"], errors="coerce")
    df = df.dropna(subset=["Work Order", "Travel Distance in KM"])
    df = df[df["Travel Distance in KM"] <= 30]

    # Load task data
    task_data = read_task_data()
    task_data = task_data[[
        "Work Order Number", "Street 1",
        "Postal Code", "City", "Latitude", "Longitude"
    ]]

    df_merged = pd.merge(df, task_data, left_on="Work Order", right_on="Work Order Number", how="left")
    df_merged = df_merged.dropna(subset=["Street 1", "Postal Code", "City", "Latitude", "Longitude"])

    # Randomly sample 1 per technician
    df_random = df_merged.groupby("Resource").apply(lambda grp: grp.sample(n=1)).reset_index(drop=True)


    df_random = df_random.rename(columns={
        "Street 1": "home_adress",
        "Postal Code": "home_postal_code",
        "City": "home_city",
        "Latitude": "home_lat",
        "Longitude": "home_long"
    })

    # Save and return
    df_random.to_excel("Data\\Aggregated\\HomeAdresses.xlsx", index=False)
    return df_random


def read_ressource_data():
    df = pd.read_excel('Data\Source\FilerV2\FSM_Active Bookable Resource Bookings_Datafil.xlsx',header=0)
    df=df[["External Order ID","Work Order","Resource","Start Time","End Time","Booking Status","Resource Requirement"]]
    df=df[df["Booking Status"]=="Udført"]
    df=df[df["External Order ID"].notnull()]
    return df

def combine_task_and_ressource():
    tasks=read_task_data()
    ressource=read_ressource_data()
    df=pd.merge(ressource,tasks,how="inner",right_on=["External Order ID","Work Order Number"],left_on=["External Order ID","Work Order"])
    funcs.calculateFTR(df) #Don't need to combine. Just calculates the seperate file.

    pd.DataFrame(df).to_excel("Data\Temp\CombinedTaskResource.xlsx")
    return df

def combine_taskressource_home_addresses(home_cities):
    df_home=read_sample_home_adresses()
    df_home=df_home[df_home["home_city"].str.lower().isin(home_cities)] 
    df_task_ressource=combine_task_and_ressource()
    df=pd.merge(df_task_ressource,df_home,how="inner",on="Resource")
    pd.DataFrame(df).to_excel("Data\Temp\CombinedTaskData.xlsx")
    return df

home_cities=["grenaa","vejle","horsens","kolding","viby j","vejle øst","esbjerg","silkeborg","middelfart"]
combine_taskressource_home_addresses(home_cities)

