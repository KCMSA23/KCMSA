
import os

# get the python path by "which python", and change the "python_dir"
python_dir = "/root/miniconda3/bin/python "

file_dir = " /KCMSA/Code/A_KAP.py "

for iter in range(5):
    parameter = "--iter " + str(iter) + " "
    cmd = python_dir + file_dir + parameter
    print(cmd)
    os.system(cmd)

res_acc = 0
res_f1 = 0
with open("result_acc.txt", "r") as f:
    lines = f.readlines()
    for line in lines[-5:]:
        line = float(line.strip("\n"))
        res_acc += line
with open("result_f1.txt", "r") as f:
    lines = f.readlines()
    for line in lines[-5:]:
        line = float(line.strip("\n"))
        res_f1 += line

print("*" * 100)
print()
print(res_acc / 5.0)
print(res_f1 / 5.0)

print("*" * 100)
