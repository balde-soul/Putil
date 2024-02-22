# coding=utf-8
import json
import sys
import os


##@brief
# @note
# @param[in]
# @param[in]
# @return 
def diff(json_first, json_second):
    return {}

def merge(json_first, json_second):
    return {}

if __name__ == '__main__':
    import argparse
    options = argparse.ArgumentParser()
    options.add_argument('files', metavar='F', type=str, nargs='+', \
        help='two json file and one target file [json_file_1, json_file2, output_json_file_to]')
    options.add_argument('--merge', action='store_true', default=False, \
        help='merge two file to one, first file is preferential')
    options.add_argument('--diff', action='store_true', default=False, \
        help='compare two json file and save the result to the output file')
    args = options.parse_args()
    result_file = args['files'][-1]
    result_dict = {'command': ''}
    if os.path.exists(result_file):
        # todo: make some message
        sys.exit()
    with open(result_file, 'w') as fp:
        # todo: do some message
        json.dump(result_dict, indent=4, check_circular=True, sort_keys=True)
        pass
    pass