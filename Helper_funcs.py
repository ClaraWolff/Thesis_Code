import pandas as pd

def calculate_ftr_per_resource(df):
    df["Created Revisit After"] = df["Created Revisit After"].astype(bool)

    # Group by Resource and calculate FTR
    ftr_by_resource = df.groupby(["Resource","Work Order Type"]).agg(
        total_tasks=('External Order ID', 'count'),
        revisit_tasks=('Created Revisit After', 'sum')
    )

    ftr_by_resource['FTR'] = (
        (ftr_by_resource['total_tasks'] - ftr_by_resource['revisit_tasks']) /
        ftr_by_resource['total_tasks']
    )

    return ftr_by_resource.reset_index()


def calculateFTR(df):
    # Make sure 'Besøg Dato' is datetime
    df['Besøg Dato'] = pd.to_datetime(df['End Time']).dt.date
    df['Created On Dato']=pd.to_datetime(df['Created On']).dt.date
    df['Adresse Fuld'] = df['Street 1'].astype(str) + ' ' + df['Postal Code'].astype(str)
    # Sort by order id, address, and date
    df = df.sort_values(by=['Adresse Fuld', 'Besøg Dato'])

    # Create helper columns
    df['Created Revisit After'] = False
    df['Is Revisit'] = False
    df['Days Since Last Visit'] = None

    # Group by address + external order id
    grouped = df.groupby(['Adresse Fuld'])
    for _,group in grouped:
        if len(group)>1:

            for i in range(1, len(group)):
                current = group.iloc[i]
                previous = group.iloc[i - 1]

                days_between = (current['Created On Dato'] - previous['Besøg Dato']).days
                created_after_visit=current['Created On']>previous['End Time']

                df_index_current = current.name
                df_index_previous = previous.name

                if days_between <= 7 and created_after_visit:
                    check='fejl' in str(current["Work Order Type"]).lower()
                    if check:
                        df.loc[[df_index_current],['Is Revisit']] = True
                        df.loc[[df_index_previous], ['Created Revisit After']] = True

                df.loc[[df_index_current], ['Days Since Last Visit']] = days_between

    df.to_excel("Data\\Temp\\Revits_detected.xlsx", index=False)
    #Aggregate on technician and task level and then compute FTR
    FTRdf=calculate_ftr_per_resource(df)
    FTRdf.to_excel("Data\\Temp\\FTRdf.xlsx", index=False)
    # Merge FTR values into the original DataFrame
    df = pd.merge(df, FTRdf, on=['Resource','Work Order Type'], how='left')
    
    df.to_excel("Data\\Temp\\FTR_per_resource_added.xlsx", index=False)

    return df

