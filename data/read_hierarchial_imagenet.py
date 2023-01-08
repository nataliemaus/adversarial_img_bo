# Imagenet classes hierarcy 
# https://observablehq.com/@mbostock/imagenet-hierarchy
def load_imagenet_hierarcy_dicts(work_dir="/home/nmaus/"):
    # path = work_dir + "adversarial_img_bo/data/image_net_hierarcy.csv"
    path = "../data/image_net_hierarcy.csv"
    f = open(path, "r") 
    l2_to_l1 = {} # 142 
    l3_to_l2 = {} 
    for line in f: 
        line = line.replace("\n", "")
        # line = line.replace(" ", "")
        line = line.split(", ")
        line = [s.strip() for s in line]
        if line[0] == "2":
            l2_to_l1[line[1]] = line[2:]
        elif line[0] == "3":
            l3_to_l2[line[1]] = line[2:]
            
            
    l3_to_l1 = {}  # 26
    for l3_class in l3_to_l2.keys():
        l2_classes = l3_to_l2[l3_class]
        all_l1_classes = []
        for l2_class in l2_classes:
            all_l1_classes = all_l1_classes + l2_to_l1[l2_class]
        l3_to_l1[l3_class] = all_l1_classes

    return l2_to_l1, l3_to_l1
