import pandas as pd

def create_event_log(run_results_path="../data/run_results.csv"):
    df = pd.read_csv(run_results_path)

    df = df[df["event_type"]=="queue"]

    df["activity_id"] = df.groupby("run_number").cumcount() + 1

    # Duplicate rows and modify them
    df_start = df.copy()

    df_start["lifecycle_id"] = "start"

    df_end = df.copy()
    df_end["lifecycle_id"] = "complete"

    # Shift timestamps for 'end' rows
    df_end["timestamp"] = df_end["timestamp"].shift(-1)
    df_end["timestamp_dt"] = df_end["timestamp_dt"].shift(-1)

    # Combine and sort
    df_combined = pd.concat([df_start, df_end]).sort_index(kind="stable")

    # Drop last 'end' row (since there’s no next row to get a timestamp from)
    df_combined = df_combined[:-1]

    df_combined.to_csv("event_log.csv", index=False)
