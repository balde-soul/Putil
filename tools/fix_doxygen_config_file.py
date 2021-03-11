# coding=utf-8
import argparse
from enum import Enum

option = argparse.ArgumentParser()
option.add_argument('--source_config', type=str, action='store', default='', \
    help='the default config file path')
option.add_argument('--target_config', type=str, action='store', default='', \
    help='the target file the save the config file fixed')

class LineType(Enum):
    empty = 0
    doc = 1
    value = 2


def line_type(line):
    if line[0] == '#':
        return LineType.doc

    elif line[0] == '\n':
        return LineType.empty
    
    else:
        return LineType.value


def get_value_name(line):
    return line.split('=')[0].replace(' ', '')


def insert_value(lines, value_index, value):
    lines.insert(value_index, value)
    return value_index + 1


if __name__ == '__main__':
    args = option.parse_args()
    assert args.source_config != ''
    assert args.target_config != ''

    with open(args.source_config, 'r') as fp:
        lines = fp.readlines()
        target_lines = []
        value_index = 1
        has_doc = False
        has_value = False
        has_empty = False
        value_name = ''
        doc_line_amount = 0
        for line in lines:
            lt = line_type(line)
            #if lt == LineType.empty:
            #    value_name = ''
            #    doc_line_amount = 0
            #    has_empty = True
            #    has_doc = False
            #    has_value = False
            #    continue

            #if lt == LineType.doc:
            #    doc_line_amount = 1 if has_empty else doc_line_amount + 1
            #    target_lines.append(line)
            #    has_empty = False
            #    has_value = False
            #    has_doc = True
            #    pass

            #if lt == LineType.value:
            #    value_name = get_value_name(line)
            #    value_index = insert_value(target_lines, value_index, line)
            #    insert_value(target_lines, len(target_lines) - doc_line_amount - 1, '#{}_DOC'.format(value_name))
            #    has_empty = False
            #    has_doc = False
            #    has_value = True
            #    pass
            if lt == LineType.empty:
                target_lines.append(line)
            if lt == LineType.doc:
                target_lines.append(line)
            if lt == LineType.value:
                target_lines.append('#{}'.format(line))
                insert_value(target_lines, value_index, line)
            pass
        with open(args.target_config, 'w') as fpw:
            fpw.writelines(target_lines)