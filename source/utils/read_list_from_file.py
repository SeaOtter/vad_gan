def read_list_from_file(filename):
    with open(filename, "r") as myfile:
        data = myfile.readlines()
        data = [ x.strip('\n \r\t') for x in data ]
        return data
    return None