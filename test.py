from tqdm.auto import tqdm
import time

for i in tqdm(range(10), desc="Testing tqdm"):
    time.sleep(0.1)