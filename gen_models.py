import os
import glob
import argparse
import collections
import copy
import csv
import functools
import gc
import io
import itertools
import logging
import os
import subprocess
import sys
from shutil import rmtree
import shlex

models = set()
folder_name = "timm_graphs"

for fn in glob.glob('docs/models/*.md'):
    with open(fn, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            if not line.startswith('model = timm.create_model('):
                continue

            model = line.split("'")[1]

            # print(model)
            models.add(model)

# print(models)
lm = list(models)
lm.sort()
# print(lm)


# with open('_models.txt', 'w') as f:
#     for model in lm:
#         f.write(model + '\n')
for name in lm:
    current_name = name
    try:
        cmd = shlex.split(f"python benchmark.py --fuser nvfuser --bench dump --model={name} -b 128")
        subprocess.check_call(cmd)
        print(name, "success")
    except subprocess.SubprocessError:
        print(name, "ERROR")
        if os.path.exists(f"{folder_name}/{current_name}"):
            rmtree(f"{folder_name}/{current_name}")
        logging.exception("removing folder")