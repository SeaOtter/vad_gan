from __future__ import division, absolute_import, print_function
import csv
import json
class ParamManager():
    def __init__(self):
        self.params = {}

    def add(self, name, value, type):
        self.params[name] = [value, type]

    def set(self, name, value, type):
        self.params[name] = [value, type]

    def clear(self):
        self.params = {}

    def get_value(self, name1, name2=None, name3=None):
        if name2 is None:
            return self.params[name1][0]
        if name3 is None:
            return self.params[name1][0], self.params[name2][0]

        return self.params[name1][0], self.params[name2][0], self.params[name3][0]
    def set_value(self, name, value):
        self.params[name][0] = value


    def get_type(self, name):
        return self.params[name][1]

    def get_all(self, type):
        D = {}
        for (name, value) in self.params:
            D[name] = value[0]
        return D

    def save_csv(self, file):
        with open(file, 'w', newline="") as f_param:
            writer = csv.writer(f_param)
            # writer.writerows(self.params.items())
            for key, value in self.params.items():
                writer.writerow([key, value[0], value[1]])

    def save_json(self, file):
        json.dump(self.params, open(file, 'w'))