import sys
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
for i in range(len(lines)-1, -1, -1):
    if lines[i].strip():
        lines[i] = lines[i].replace('run_mapping_mission()', 'run_frontier_exploration()\n')
        break
with open(sys.argv[1], 'w') as f:
    f.writelines(lines)
print('OK')
