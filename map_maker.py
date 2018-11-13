# !/usr/local/bin/python3

import csv, json

fname = 'hi_state_id_island_code_map.csv'
with open(fname) as filob:
	reader = csv.reader(filob)
	frame = [row for row in reader]
mapper = {}
for row in frame:
	name, stid, islcode = row[0], row[1], row[2]
	if not all([i=='' for i in (stid, islcode)]):
		mapper[stid] = islcode
	else:
		print('no values for {}'.format(name))
json_string = json.dumps(mapper)
with open('hi_island_code_map.json', 'w') as mapfile:
	mapfile.write(json_string)
print('Wrote a json dict file with {} entries'.format(len(mapper)))



