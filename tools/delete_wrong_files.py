import os
import time
import shutil
import random
import csv
import numpy as np
from dotenv import load_dotenv
load_dotenv()

current_time = time.time()

PAINDICATOR_RESULTS = os.getenv("PAINDICATOR_RESULTS")
print PAINDICATOR_RESULTS

folders = [os.path.join(PAINDICATOR_RESULTS, f) for f in os.listdir(PAINDICATOR_RESULTS)]

for folder in folders:
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    # print files
    for f in files:
        creation_time = os.path.getctime(f)
        print (current_time - creation_time) // (24 * 3600)
        if (current_time - creation_time) // (24 * 3600) <= 5:
            # print f
            os.unlink(f)
            print('{} removed'.format(f))