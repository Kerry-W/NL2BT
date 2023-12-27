import xml.etree.ElementTree as ET
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import time

class Stack(object):

    def __init__(self):
         self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)


def prettyXml(element, indent, newline, level = 0):
    if element:
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    temp = list(element) 
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else: 
            subelement.tail = newline + indent * level   
        prettyXml(subelement, indent, newline, level = level + 1)

    
def write_tree(teaching_name, teaching_param, teaching_root, main_seq_n):
    save_root = teaching_root.pop()
    now_param = teaching_param.peek() 
    if teaching_root.is_empty(): # main_tree 
        seq_node = main_seq_n
    else:
        seq_node = teaching_root.peek()   
    temp_dict = {}
    temp_dict['ID'] = teaching_name.peek()
    if temp_dict['ID'] == '移动':  
        temp_dict['obj'] = dic_param['des']
    else:
        for p in now_param:
            temp_dict[p] = now_param[p]

    output_keys = []
    all_bb = seq_node.findall('SetBlackboard')
    for i in all_bb:
        output_keys.append(i.attrib['output_key'])
    for k, v in temp_dict.items():
        if k != 'ID':
            if v not in output_keys:
                ET.SubElement(seq_node, 'SetBlackboard', attrib={'output_key': v, 'value': v})
    
    temp_node = ET.SubElement(seq_node, 'SubTree', attrib=temp_dict)
    teaching_root.push(save_root) 

def new_skill(skill_name,intent, dic_param, tree,root,index):
    for BehaviorTree in root.findall('BehaviorTree'):
        if BehaviorTree.attrib['ID'] == 'BehaviorTree':
            mainTree_n = BehaviorTree
            main_seq_n = BehaviorTree.find('Sequence')

    # 3种意图分类
    if intent=='1': # teaching
        teaching_name.push(skill_name)
        teaching_param.push(dic_param) 

        new_sub_tree = ET.SubElement(root,'BehaviorTree', attrib={'ID':teaching_name.peek()})
        seq_node = ET.SubElement(new_sub_tree,'Sequence')
        for p in dic_param: 
            bb_node = ET.SubElement(seq_node,'SetBlackboard', attrib={'output_key':dic_param[p],'value':'{'+p+'}'})
        teaching_root.push(seq_node)

        write_tree(teaching_name, teaching_param, teaching_root, main_seq_n)

    elif intent=='2': # instruction
        if teaching_root.is_empty(): 
            seq_node = main_seq_n
        else:
            seq_node = teaching_root.peek()
        temp_dict = {}
        temp_dict['ID'] = skill_name
        if skill_name == '移动':  
            temp_dict['obj'] = dic_param['des']
        else:
            for p in dic_param:
                temp_dict[p] = dic_param[p]
        
        output_keys = []
        all_bb = seq_node.findall('SetBlackboard')
        for i in all_bb:
            output_keys.append(i.attrib['output_key'])
        for k, v in temp_dict.items():
            if k != 'ID':
                if v not in output_keys:
                    ET.SubElement(seq_node, 'SetBlackboard', attrib={'output_key': v, 'value': v})

        temp_node = ET.SubElement(seq_node, 'SubTree', attrib=temp_dict)

        
    elif intent=='3': # completion
        teaching_root.pop()
        teaching_name.pop()
        teaching_param.pop()



def main_func(input_context):
    intent,skill_name,*param = input_context.split(' ')
    param_len = len(param)
    i_tmp = 0
    dic_param = {}
    while i_tmp<param_len-1:
        key = param[i_tmp]
        value = param[i_tmp+1]
        dic_param[key] = value
        i_tmp += 2
    new_skill(skill_name,intent,dic_param,tree,root,index)
    prettyXml(root, '\t', '\n') 
    tree.write(path, encoding="utf-8", xml_declaration=True)



path = r'~/BTScpp/BehaviorTree.CPP/build/examples/myTree.xml'
tree = ET.parse(path)
root = tree.getroot()
for BehaviorTree in root.findall('BehaviorTree'):
    if BehaviorTree.attrib['ID'] == 'BehaviorTree':
        mainTree = BehaviorTree
        main_seq = BehaviorTree.find('Sequence')
        temp_list = [all_children for all_children in main_seq]
        for all_children in temp_list:
            main_seq.remove(all_children)
prettyXml(root, '\t', '\n') 
tree.write(path, encoding="utf-8", xml_declaration=True)
tree = ET.parse(path)
root = tree.getroot()
teaching_name = Stack()
teaching_root = Stack() # sequence node
teaching_param = Stack()

index = 1 

post_process_file = "~/flask/PostProcess.txt"
mtime = os.path.getmtime(post_process_file)  
while True:
    last_mtime = mtime
    mtime = os.path.getmtime(post_process_file)
    if mtime != last_mtime:
        with open (post_process_file, 'r') as f:
            lines = f.readlines()
            input_text_t = lines[-1]  
            input_text = input_text_t[:-1]  
        if input_text == "quit":
            break
        else:
            print(input_text)
            main_func(input_text)

            ## for real-time execution:
            # if input_text.split(' ')[0] == '2':
            #     os.system('cd ~/BTScpp/BehaviorTree.CPP/build/examples && ./Grab')
        