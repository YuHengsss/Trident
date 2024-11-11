import os
configs_list = [
    './configs/cfg_ade20k.py',
    './configs/cfg_city_scapes.py',
    './configs/cfg_coco_object.py',
    './configs/cfg_coco_stuff164k.py',
    './configs/cfg_context59.py',
    './configs/cfg_context60.py',
    './configs/cfg_voc20.py',
    './configs/cfg_voc21.py'
    # './configs/cfg_context459.py',
    # './configs/cfg_ade847.py',
]

for config in configs_list:
    print(f"Running {config}")
    os.system(f"bash ./dist_test.sh {config}")
