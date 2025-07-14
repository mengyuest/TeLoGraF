from generate_scene_v1 import SimpleAnd, SimpleOr, SimpleListAnd, SimpleListOr, SimpleNot, SimpleF, SimpleG, SimpleUntil, SimpleReach, check_stl_type
import random
import copy
import numpy as np

def rand_aug(node, inplace=True):
    if inplace==False:
        node = copy.deepcopy(node)
    if isinstance(node, SimpleAnd) or isinstance(node, SimpleOr):
        random.shuffle(node.children)
    elif isinstance(node, SimpleListAnd) or isinstance(node, SimpleListOr):
        random.shuffle(node.children)
    for child in node.children:
        rand_aug(child)
    
    if inplace==False:
        return node

def hard_rand_aug(node, cfg, father=None, grandfather=None, inplace=True):
    if inplace==False:
        node = copy.deepcopy(node)
    if isinstance(node, SimpleReach):
        if isinstance(father, SimpleNot):
            if isinstance(grandfather, SimpleG):
                node.obj_x = node.obj_x + np.random.uniform(-0.2, 0.2) * node.obj_r
                node.obj_y = node.obj_y + np.random.uniform(-0.2, 0.2) * node.obj_r
                node.obj_r = node.obj_r * np.random.uniform(0.8, 1.0)
            elif isinstance(grandfather, SimpleUntil):
                node.obj_x = node.obj_x + np.random.uniform(-0.2, 0.2) * node.obj_r
                node.obj_y = node.obj_y + np.random.uniform(-0.2, 0.2) * node.obj_r
                node.obj_r = node.obj_r * np.random.uniform(1.0, 1.2)
            else:
                raise NotImplementedError
        else:
            node.obj_x = node.obj_x + np.random.uniform(-0.2, 0.2) * node.obj_r
            node.obj_y = node.obj_y + np.random.uniform(-0.2, 0.2) * node.obj_r
            node.obj_r = node.obj_r * np.random.uniform(1.0, 1.2)
    elif isinstance(node, SimpleF):
        node.te = max(0, min(cfg["tmax"], np.random.randint(node.te-5, node.te+5)))
        node.ts = max(node.te+1, max(0, min(cfg["tmax"], np.random.randint(node.ts-5, node.ts+5))))
    elif isinstance(node, SimpleAnd):
        if len(node.children)>2:
            left_cnt = np.random.randint(2, len(node.children))
            choices = np.random.choice(len(node.children), left_cnt, replace=False)
            node.children = [node.children[choice] for choice in choices]
    
    for child in node.children:
        hard_rand_aug(child, cfg, node, father)
    
    if inplace==False:
        return node

def compute_tree_size(node, cnt_d=None):
    cnt_d["n"] += len(node.children)
    for child in node.children:
        compute_tree_size(child, cnt_d=cnt_d)


def aug_graph(node, cfg, father=None, grandfather=None, inplace=True, max_aug=None, curr_cnt_stat=None):
    if inplace==False:
        node = copy.deepcopy(node)
    if isinstance(node, SimpleReach):
        do_nothing=True
    elif isinstance(node, SimpleF):
        do_nothing=True
    elif isinstance(node, SimpleAnd) or isinstance(node, SimpleOr) or isinstance(node, SimpleListAnd) or isinstance(node, SimpleListOr):
        if max_aug>=1:
            left_cnt = np.random.randint(1, max_aug+1)
            terminal_nodes = [val_node for val_node in node.children if (isinstance(val_node, SimpleF) or isinstance(val_node, SimpleG))]
            if len(terminal_nodes)>0:
                choices = np.random.choice(len(terminal_nodes), left_cnt, replace=True)
                for choice_i in choices:
                    node.children.append(copy.deepcopy(terminal_nodes[choice_i]))
                    cnt_d = {"n":1}
                    compute_tree_size(terminal_nodes[choice_i], cnt_d=cnt_d)
                    curr_cnt_stat["add"] += cnt_d["n"]
            else:
                choices = []
        else:
            terminal_nodes = []
            choices = []
        
    for child in node.children:
        aug_graph(child, cfg, node, father, max_aug=max_aug, curr_cnt_stat=curr_cnt_stat)
    
    if inplace==False:
        return node
    

def stl_naive_str(obj_dict_d, real_stl):
    obj_str = " ".join(["%d,%.4f,%.4f,%.4f,%.4f"%(objid, objval["x"], objval["y"], objval['z'], objval["r"]) for objid,objval in obj_dict_d.items()])
    stl_str = str(real_stl)
    merged_str = (stl_str + "#" + obj_str).replace(" ", "")
    merged_str = [ord(xxx) for xxx in merged_str]
    return merged_str

def stl_smart_encode(stl_lines, pad_len):
    new_array2=[]
    for xxx in stl_lines:
        pad_array = [-1] * pad_len
        pad_array[1:len(xxx)] = xxx[1:]
        new_array2.append(pad_array)
    return new_array2

def stl_to_1d_array(stl_lines):
    new_array = [[-1]+xxx[1:] for xxx in stl_lines]
    new_array2 = []
    for ele in new_array:
        new_array2 += ele
    return new_array2

def stl_to_seq(stl_tree, is_3d=False):
    seq = []
    queue = [(0, stl_tree)]
    while len(queue) != 0:
        depth, node = queue[0]
        del queue[0]
        #print(node)
        node_type_i = check_stl_type(node)
        ts, te = node.ts, node.te
        if ts is None:
            ts = -1
            te = -1
        x, y, z, r = 0, 0, 0, -1
        if isinstance(node, SimpleReach):
            if is_3d:
                x, y, z, r = node.obj_x, node.obj_y, node.obj_z, node.obj_r
            else:
                x, y, r = node.obj_x, node.obj_y, node.obj_r
        n_child = len(node.children)
        seq.append([node_type_i, ts, te, x, y, z, r, n_child])
        for child in node.children:
            queue.append((depth+1, child))
    return seq

def stl_hash_seq(stl_tree, is_3d=False, is_root=True):
    CB = {
        2:   tuple([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  negation
        0:   tuple([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  conjunction
        1:   tuple([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  disjunction
        5:   tuple([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  eventually
        6:   tuple([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  always
        7:   tuple([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  #  until
        "(": tuple([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  #  left bracket
        ")": tuple([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  #  right bracket
        "t": tuple([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),  #  time
        8:   tuple([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),  #  reach
    }
    LEFT_IDX=8
    RIGHT_IDX=9
    node_type_i = check_stl_type(stl_tree)
    code = list(CB[node_type_i])
    
    if node_type_i == 8:
        if is_3d:
            code[-4:] = [stl_tree.obj_x, stl_tree.obj_y, stl_tree.obj_z, stl_tree.obj_r]
        else:
            code[-4:] = [stl_tree.obj_x, stl_tree.obj_y, 0, stl_tree.obj_r]
        return [code]
    
    elif node_type_i in [0, 1]: # (A) AND/OR (B) AND/OR (C) ...
        code_list = []
        for child_i, child in enumerate(stl_tree.children):
            child_code = stl_hash_seq(child, is_3d=is_3d, is_root=False)
            if child_i==0:
                code_list = [list(CB["("])] + child_code + [list(CB[")"])]
            else:
                code_list = code_list + [code] + [list(CB["("])] + child_code + [list(CB[")"])]
        return code_list
    
    elif node_type_i in [2]: # NOT
        child1_code = stl_hash_seq(stl_tree.children[0], is_3d=is_3d, is_root=False)
        return [code] + [list(CB["("])] + child1_code + [list(CB[")"])]
    
    elif node_type_i in [7]: # until
        time_code = list(CB["t"])
        time_code[LEFT_IDX] = stl_tree.ts
        time_code[RIGHT_IDX] = stl_tree.te
        if is_3d:
            time_code[LEFT_IDX] = time_code[LEFT_IDX] / 64
            time_code[RIGHT_IDX] = time_code[RIGHT_IDX] / 64
        code_list = []
        # assert len(stl_tree.children)==2
        child1_code = stl_hash_seq(stl_tree.children[0], is_3d=is_3d, is_root=False)
        child2_code = stl_hash_seq(stl_tree.children[1], is_3d=is_3d, is_root=False)
        return [list(CB["("])] + child1_code + [list(CB[")"])] + [code, time_code] + [list(CB["("])] + child2_code + [list(CB[")"])]
    
    elif node_type_i in [5, 6]: # eventually/always
        time_code = list(CB["t"])
        time_code[LEFT_IDX] = stl_tree.ts
        time_code[RIGHT_IDX] = stl_tree.te
        if is_3d:
            time_code[LEFT_IDX] = time_code[LEFT_IDX] / 64
            time_code[RIGHT_IDX] = time_code[RIGHT_IDX] / 64
        child1_code = stl_hash_seq(stl_tree.children[0], is_3d=is_3d, is_root=False)
        return [code, time_code] + [list(CB["("])] + child1_code + [list(CB[")"])]
    
    else:
        raise NotImplementedError

def parse_str(stl_str_list):
    LEFT_IDX=8
    RIGHT_IDX=9
    CB = {
        2:   tuple([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  negation
        0:   tuple([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  conjunction
        1:   tuple([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  disjunction
        5:   tuple([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  eventually
        6:   tuple([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  #  always
        7:   tuple([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  #  until
        "(": tuple([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  #  left bracket
        ")": tuple([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  #  right bracket
        "t": tuple([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),  #  time
        8:   tuple([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),  #  reach
    }
    symbol_d = {
        2: "Not",
        0: "And",
        1: "Or",
        5: "F",
        6: "G",
        7: "U",
        "(": "(",
        ")": ")",
    }
    inv_CB = {v:k for k,v in CB.items()}
    s=""
    for word_ in stl_str_list:
        word = tuple(word_)
        if word in inv_CB and inv_CB[word] not in ["t", 8]:
            s += symbol_d[inv_CB[word]]
        else:
            if word[LEFT_IDX]!=0 or word[RIGHT_IDX]!=0: # time
                s += "[%d,%d]"%(word[LEFT_IDX], word[RIGHT_IDX])
            else: # reach
                s += "Reach{%.1f,%.1f,%.1f,%.1f}"%(word[-4],word[-3],word[-2],word[-1])
    return s


def main():
    reach = SimpleF(0, 5, SimpleReach(obj_id=8, object=[0, 0, 1.0]))
    
    or_reach = SimpleOr(
        SimpleF(0, 5, SimpleReach(obj_id=6, object=[0, 0, 1.0])),
        SimpleF(0, 8, SimpleReach(obj_id=7, object=[2, 3, 1.0]))
    )
    
    seq_reach = SimpleF(2, 12, SimpleAnd(SimpleG(0, 5, SimpleReach(obj_id=4, object=[0.5, 0.2, 0.1])), SimpleF(1, 5, SimpleReach(obj_id=5, object=[5, 2, 1]))))
    
    untils = SimpleUntil(0, 15, SimpleNot(SimpleReach(obj_id=2, object=[0, 0, 1.0])), SimpleReach(obj_id=3, object=[-2.0, -2.5, 0.5]))
    
    avoid0 = SimpleG(0, 10, SimpleNot(SimpleReach(obj_id=0, object=[1.0, 2.0, 2.5])))
    avoid1 = SimpleG(5, 18, SimpleNot(SimpleReach(obj_id=1, object=[-1.0, 0.5, 0.5])))
    
    a_bunch_of_stls = [or_reach, seq_reach, untils, avoid0, avoid1]
    # a_bunch_of_stls = [reach, avoid0]
    
    stl_tree = SimpleListAnd(a_bunch_of_stls)
    
    print("stl_tree")
    stl_tree.print_out()
    print()
    
    stl_tree1 = rand_aug(stl_tree, inplace=False)
    
    print("after rand_aug, stl_tree1")
    stl_tree1.print_out()
    print()
    
    print("after rand_aug, the old stl_tree")
    stl_tree.print_out()
    print()
    
    
    rand_aug(stl_tree1)
    print("after 2nd rand_aug, stl_tree1")
    stl_tree1.print_out()
    
    print("after rand_aug, the old stl_tree")
    stl_tree.print_out()
    print()
    return

if __name__ == "__main__":
    main()