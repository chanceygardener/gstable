#!/usr/local/bin/python3

def isfloat(nstr):
	nstr = str(nstr) if not isinstance(nstr, str) else nstr
	one_dec = (nstr.count('.') <= 1)
	numer = (nstr.replace('.', '').isnumeric())
	return True if one_dec and numer else False
    

def normalize(string, strip_zeroes=True):
	output = string.lower().strip()
	output = output.replace(' ', '_')
	output = output.replace('/', '_')
	output = output.replace(', ', '_')
	output = output.replace(',', '_')
	output = output.replace(':', '-')
	output = output.replace('__', '_')
	output = output.replace('"', '')
	if strip_zeroes:
		output = output.lstrip('0')
	return output

def make_table(filepath, headed=True, skip_columns=False, delimiter='\t'):
    outtakes = []
    filob = open(filepath, 'r')
    table = [line.split(delimiter) for line in filob.read().split('\n')]
    table = [r for r in table if not (len(r) == 1 and r[0]=='')]
    if skip_columns:
    	#print('\nskip columns parameter was recognized!!!\n')
    	use_columns = []
    	skip_columns = [normalize(i) for i in skip_columns]
    	headers = [normalize(head) for head in table[0]]
    	for ind, head in enumerate(headers):
    		if head not in skip_columns:
    			use_columns.append(ind)
    		else: # testing
    			print('COLUMN {} was SKIPPED'.format(head))
    	table = [[row[ind] for ind in range(len(headers)) if ind in use_columns] for row in table]
    return table