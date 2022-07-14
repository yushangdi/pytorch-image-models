import os
import glob
import logging
import os
import subprocess
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

batch_size_dict = {
    "dpn107": 64,
    "gluon_senet154": 64,
    "gluon_xception65": 64,
    "ig_resnext101_32x16d": 64,
    "legacy_senet154": 64,
    "nasnetalarge": 64,
    "pnasnet5large": 64,
    "resnetv2_101x1_bitm": 64,
    "seresnet152d": 64,
    "ssl_resnext101_32x16d": 64,

}

SKIP = {
    "tresnet_l"
}

# with open('_models.txt', 'w') as f:
#     for model in lm:
#         f.write(model + '\n')
for name in lm:
    current_name = name
    if name in SKIP:
        continue
    try:
        batch_size = batch_size_dict.get(name, 128)
        cmd = f"python benchmark.py --fuser nvfuser --bench dump --model={name} -b {batch_size}"
        print(cmd)
        cmd = shlex.split(cmd)
        subprocess.check_call(cmd)
        print(name, "success")
    except subprocess.SubprocessError:
        print(name, "ERROR")
        if os.path.exists(f"{folder_name}/{current_name}"):
            rmtree(f"{folder_name}/{current_name}")
        logging.exception("removing folder")