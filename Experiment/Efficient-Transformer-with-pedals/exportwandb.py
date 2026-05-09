from pathlib import Path

import wandb
import pandas as pd

api = wandb.Api()
run = api.run("jnneil10-national-dong-hwa-university/AMT-audio-to-midi/2m2ag07k")
output_path = Path(__file__).with_name("run_2m2ag07k_history.csv")

df = pd.DataFrame(run.scan_history())
df.to_csv(output_path, index=False)

print(f"Exported {len(df)} rows to {output_path}")
