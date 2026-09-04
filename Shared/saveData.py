import pandas as pd
import os
from datetime import datetime

MASTER_FILE = "all_wells.csv"

def append_plate_to_csv(Fluostar_data, Roi_df, Master_file=MASTER_FILE):
    rows = []
    for i in range(len(Roi_df)):
        coord = Roi_df['well'].iloc[i]
        row = {
            "fileFolder": Fluostar_data.FileFolder,
            "fileName": Fluostar_data.FileName,
            "well": coord,
            "condition": Fluostar_data.dict.get(coord, ""),
            "nFibril": Roi_df['nFibril'].iloc[i],
            "nBase": Roi_df['nBase'].iloc[i],
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        rows.append(row)
    
    new_df = pd.DataFrame(rows)

    # Append or create
    if os.path.exists(Master_file):
        old_df = pd.read_csv(Master_file)
        df_combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df_combined = new_df
    
    df_combined.to_csv(Master_file, index=False)
    print(f"Appended {len(new_df)} wells to {Master_file}")

# Usage
# append_plate_to_csv(data, roi)
