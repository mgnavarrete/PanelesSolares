def get_all_class(data, label):
    '''
    Find images that have specific object
    :param data: data dictionary with data
    :param label: label code, ie: 0,1,3,...
    :return: list of path of images with specific class
    '''
    list_of_path = set()
    for key, values in data.items():
        for v in values:
            if v['obj_class'] == label:
                list_of_path.add(key)
    return list(list_of_path)


def count_class(data, label):
    '''
    Find images that have specific object
    :param data: data dictionary with data
    :param label: label code, ie: 0,1,3,...
    :return: number of specificar object in the dataset
    '''
    list_of_path = list()
    for key, values in data.items():
        for v in values:
            if v['obj_class'] == label:
                list_of_path.append(key)
    return len(list_of_path)




