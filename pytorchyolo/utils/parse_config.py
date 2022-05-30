

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):  # 这里传进来的是.data这个文件的路径
    """Parses the data configuration file"""
    options = dict()  # 这里设置了一个选项，其中包括使用gpu的数量
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:  # 打开.data文件，创建一个对象
        lines = fp.readlines()  # 把.data中的每一行进行读入，lines是一个列表，其中的每一个元素是.data文件中的每一行（行尾有\n）
    for line in lines:  # 遍历lines列表中的每一个元素
        line = line.strip()  # 把行尾的"\n"去掉
        if line == '' or line.startswith('#'):  # 如果要是行中没有信息的话就不往下执行了
            continue
        key, value = line.split('=')  # 把key和value分开
        options[key.strip()] = value.strip()  # 如果要是等号周围有空格的话，进行去掉。然后一把key和value对应起来放在字典中
    return options  # 返回一个字典，其中的信息有gpus、num_workers、还有.data文件中的各种信息

    fp.close()
