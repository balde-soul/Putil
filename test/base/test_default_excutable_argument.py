# coding=utf-8
import base.default_excutable_argument as dea

argument = dea.Argument()
print(argument.log_file_path)
print(argument.log_file_dir)
print(argument.log_file_name)
args = argument.parser.parse_args()
