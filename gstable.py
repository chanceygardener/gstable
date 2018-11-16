#!//Library/Frameworks/Python.framework/Versions/3.7/bin/python3

from copy import deepcopy as dpc
import re, json, os


# Helper functions for GsTable class

uname = 'ibicket'
# file map init @ 487

fmap_template = {'state_id': 'state', 
							'grade':'all',
							'entity_name':None,
							'subject':None,
							'entity_type': None, 
							'breakdown':'all',
							'data_types': None,
							'target': ''
							}


def run(script_name='processor.py'):
	'''run your processor script from the console to explore le data'''
	with open(script_name) as script_file:
		script = script_file.read()
	exec(script)
	return load

def alph_to_base_10(astr):
	'''given a string in the format of MS excel column headers, 
	return the index of that column in base 10 [ind starts at 0]'''
	icount = 0
	chlist = [(ord(char.lower())-ord('a')+1) for char in astr]
	mcount = len(chlist)
	for ind, ch in enumerate(chlist):
		ind += 1
		coef = 26**(mcount-ind)
		icount += ch*coef
	icount -= 1
	return icount



def congen(p):
	if not isinstance(p, bool):
		raise ValueError("condition generator takes a boolean argument")
	return lambda arg: True if p else False 


def tupind(arr, inds):
	return tuple([arr[i] for i in inds])

def read_qconfig(qname, queue_file_delimiter=':queue_modifier:'):
	with open(qname) as qfile:
		qdat = qfile.read().split('\n')
		qdat = [i for i in qdat if i != '']
		qdat = [entry for entry in [i.strip().split(queue_file_delimiter) for i in qdat if i != ''] if entry != '']
		qstruct = [{line[j]:line[j+1] for j in range(0, len(line)-1, 2)} for line in qdat]
	return qstruct

def display_qconfig(fname):
	struct = read_qconfig(fname)
	schools = [school for school in struct if school['school_or_district'].strip()=='school']
	districts = [dist for dist in struct if dist['school_or_district'] == 'district']
	unmtchd = {'school':schools, 'district':districts}
	queries={}
	# display results
	for etype in unmtchd:
		print('State IDs for the following {} {}s were not recognized by our database:\n'.format(len(unmtchd[etype]), etype))
		for entity in unmtchd[etype]:
			print('\t* | {} | State ID: {}'.format(entity['name'], entity['state_id']))

def safe_proportion(num, denom, round_to=3):
	global err_val_count
	if not (isfloat(num) and isfloat(denom)):
		output = ''
		return output
	num, denom = float(num), float(denom)
	if denom == 0 and num == 0:
		output =  '0'
	elif num > denom:
		print('NUMERATOR: {} GREATER THAN DENOMINATOR: {}'.format(num, denom))
		err_val_count += 1
		output =  'ERR'
	else:
		output = str(round(num/denom, round_to))

	return output

def build_crosshash(fname, kcol=0, vcol=1, headed=True, delim='\t'):
	'''CSV MODE: input is csv hash w/keys in column1, vals in column2 file return a python dictionary
	while throwing warnings in the case of duplicate keys
	
	param:kcol: column in crosswalk to be used as key for crosshash object
	this should have no duplicates, but the function will check for them
	
	param:vcol: column intended to be used as value

	'''
	infile = open(fname)
	dat = infile.read()
	infile.close()
	grid = [row.split(delim) for row in dat.split('\n')]
	# get headers
	if headed:
		headers = grid.pop(0)
		print(headers)
		print('\n\ncrosswalk hash built from {}:\nkeys: {}\nvalues: {}\n\n'.format(fname, headers[kcol], headers[vcol]))
	# check for duplicates:
	keys = [row[kcol] for row in grid] if isinstance(kcol, int) else tupind(row, kcol)
	if hasdupes(keys): # checks  to see if there are duplicate keys before iterating over them
		# check build counter for unique values in xcol
		checker = {kval:[ind for ind, k in enumerate(keys) if k == kval] for kval in keys}
		# filter counter for duplicates only
		checker = {kval:checker[kval] for kval in checker if len(checker[kval])>1}
		for dupe in checker: # iterate over duplicates to print warnings
			dinds = ', '.join(dupe)
			print("WARNING! duplicate key {} in crosswalk at rows {}".format(dupe))
	out = {row[0]: row[1] for row in grid}
	return out


def isfloat(nstr):
	nstr = str(nstr) if not isinstance(nstr, str) else nstr
	one_dec = (nstr.count('.') <= 1)
	numer = (nstr.replace('.', '').isnumeric())
	return True if one_dec and numer else False

def hasdupes(arr):
	return True if len(set(arr)) == len(arr) else False

def get_duplicate_values(inhash):
	'''given a hash of the form {CODE:[(ind1,(val1, val2, val3))]}
	return a list of ind tuples representing duplicate val tuples 
	'''
	output = []
	for code in inhash:
		dupe_buff = []
		for ind, prfl in inhash[code]:
			if prfl not in dupe_buff:
				#print(prfl)
				dupe_buff.append(prfl)
			else:
				dupe = tuple(idx[0] for idx in inhash[code] if idx[1]==prfl)
				print('STATE_ID: {} | duplicated at rows {}'.format(code, str(dupe)[1:-1]))
				output.append(dupe)
	return output

def valid_grade(val):
	return True if str(val) in ['1', '2', '3', '4', '5', '6', '7',
								'8', '9', '10', '11', '12', 'UG', 'K', 'PK', 'all'] else False

def valid_subj(val, map_fname='/Users/ibicket/data_studio/gstable/subjects.json'):
	mapper = map_reader(map_fname)
	return True if val in mapper.keys() else False

def valid_breakdown(val, map_fname='/Users/ibicket/data_studio/gstable/census_breakdowns.json'):
	mapper = map_reader(map_fname)
	return True if val in mapper.keys() else False

def to_perl_hash(inhash):
	output = str(inhash).replace(':', ' =>').replace(',', ',\n')
	return output

def get_census_data_types(inval, map_fname='/Users/ibicket/data_studio/gstable/census_data_types.json', string_only=False):
	data_type_mapper = map_reader(map_fname, integer_keys=True)
	if type(inval) in (list, tuple):
		output = {data_type_mapper[val]:val for val in inval}
	else:
		if string_only:
			output = data_type_mapper[inval]
		else:
			output = {data_type_mapper[inval]:inval}

	return output


def outgen(out, ticket):
	return ticket+'_'+out

def remove_non_numeric_chars(nstr):
	return ''.join([i for i in nstr if not i.isnumeric()])


def num_norm(nstr, null_val='***'):
	out = nstr
	out = out.replace('"', '')
	out = out.replace(',', '')
	out = out.replace("'", '')
	out = out.replace('%', '')
	if out == null_val:
		out = ''
	return out

def map_reader(mpath, integer_keys=False):
	'''given a json filepath, return a hash'''
	infile = open(mpath, 'r')
	mapobj = json.loads(infile.read())
	infile.close()
	if integer_keys:
		mapobj = {int(ikey):mapobj[ikey] for ikey in mapobj}
	return mapobj

def map_to_new(fpath, data_table, column_name):
	'''given a csv file with 2 columns
	representing parts of a code to be combined,
	return a hash table with keys representing an old value
	and values representing the new value to be replaced'''
	islmap = map_reader(fpath)
	full_codes = set()
	anomalies = set()	
	success_count = 0
	schind = data_table.colhash[column_name]
	for ind, code in enumerate(data_table[column_name]):
		try:
			if islmap[code] not in full_codes:
				new_code = islmap[code]+'-'+code
				islmap[code]=new_code
				full_codes.add(new_code)
			#tb['state_id'][ind] = new_code
			success_count += 1
		except KeyError:
			print('School with id: {} has no island code specified by mapper; check the database'.format(code))
			anomalies.add(code)
	print('mapped {} school codes successfully with {} value(s) missing from mapper'.format(success_count, len(anomalies)))
	print('Successful Codes:')
#for entry in full_codes:
#		print(entry)
	return islmap


def hash_replace(inval, table, verbose=False):
	'''given value and a hash table, return the value mapped to
	the key that matches the input value, otherwise, just return
	the input value.'''
	try:
		out = table[inval]
		if verbose:
			print('{} replaced with {}'.format(inval, out))
	except KeyError:
		out = inval
		if verbose:
			print('Key Error in {} for {}'.format(inval))
	return out

def pull(targ_path, usr=uname, svr='datadev'):
	if targ_path[0] == '/':
		targ_path = targ_path[1:]
	command = 'scp {}@{}:/{} .'.format(usr, svr, targ_path)
	try:
		os.system(command)
		print('successfully retrived {}'.format(targ_path))
		return True
	except OSError as e:
		print('SCP failed, error report:\n{}'.format(e))
		return False


def push(dest, infile, usr, svr):
	if dest[0] == '/':
		dest = dest[1:]
	command = 'scp {} {}@{}:/{}'.format(infile, usr,
										 svr, dest)
	try:
		os.system(command)
		print('successfully copied {} to {} at {}'.format(infile, dest,svr))
		return True
	except OSError as e:
		print('SCP failed, error report:\n{}'.format(e))
		return False


	
#TODO: allow for list arg to const_column

def idpad_concat(row, columns, pad_right=False, pad_left=False):
	'''Given a data row, return a concatenation of several of its columns

		NOTE: columns parameter should be a list of tuples of the following form
		(col_ind, padlength)'''

	#output = []
	# make a tab
	if not all([type(i)==tuple for i in columns]):
		pad_lengths = [max(val, key=len(str(val))) for val in columns]
		arg_row = columns
	else:
		arg_row = [row[i[0]] for i in columns] # array of only relevant row values
		pad_lengths = [c[1] for c in columns]
	new_val = ''
	if pad_left:
			new_val += '0'*pad_left
	for arg_row_ind, val in enumerate(arg_row):
		pl = pad_lengths[arg_row_ind]
		padded = val.zfill(pl)
		new_val += padded
	if pad_right:
		new_val += '0'*pad_right
	
	#output.append(new_val)
	return new_val

def truncate(string, cutsize=3, right=True):
	
	if right:
		ind = -cutsize
		output = string[:ind]
	else:
		ind = cutsize
		output = string[ind:]
	return output

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


def tonum(n):
    try:
        n = float(n)
    except ValueError:
        return None
    if n % 1 == 0:
        out = int(n)
    else:
        out = float(n)
    return out

def iswholenum(nstr):
	return True if float(nstr)%1 == 0 else False

def isnum(row, value, column):
	return True if row[column].isnumeric() else False

def colis(row, value, column): 
	if type(value) in (list, tuple):
		cnd = True if row[column].strip() in value else False
	else:
		return True if row[column].strip() == value else False

def generate_column_index(table):
	'''return a dictionary mapping column header keys to their numerical
	indexes in the table'''
	return {col: ind for ind, col in enumerate(table[0])}

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

# def normalize(string):
# 	'''return snake cased string'''
# 	return string.lower().replace(' ', '_').replace('/', '_')

def tonum(n):
    try:
        n = float(n)
    except ValueError:
        return None
    if n % 1 == 0:
        out = int(n)
    else:
        out = float(n)
    return out

def iswholenum(nstr):
	return True if float(nstr) % 1 == 0 else False


def proportion(table, columns):
	''' Takes a proportion of columns[0]/columns[1]
	for each row in the table and returns the resulting array of floats 
	'''
	for col in columns:
		if not all([val.isnumeric() for val in table[col]]):
			raise ValueError("Not all values in the {} column are numeric...".format(col))
	if len(columns) != 2:
		raise ValueError("I can't take a proportion of {} values...".format(len(columns)))
	targ_ind, pos_ind = table.colhash[columns[0]], table.colhash[columns[1]]
	new = [[row[targ_ind]/row[pos_ind]] for row in table]
	return new




	if not all([type(i) in (int, float) for i in (target, possible)]):
		target, possible = float(target), float(possible)
	if target > possible:
		print('WARNING: proportion greater than 1, check your input')
	prop = target/possible
	return prop

perl_script_outer_frame = """#!/usr/bin/perl
require '/home/scripts/data/census_loader.pl';

my $STATE = '{}'; # Lower case please
my $YEAR = '{}'; # Leave blank if you are going to map a year column and get the year(s) from the file.
my $SOURCE_ID = {}; # From gs_schooldb.DataSource
my $LOAD_NAME = '{}';
my @SCHOOL_TYPES = ('public','charter','private');
my $UNIQUE_ID_TYPE = '{}';
my $CREATE_CLEAN_SLATE = 1;

## Create/find data type variables
my @DATATYPES = (
  {} 
);

## Map the layout of the files
my $FILE_LAYOUT = {{
    {}
	
}};

my $QA_ONLY = {};
load_census_data($STATE,$YEAR,$SOURCE_ID,$LOAD_NAME,\@SCHOOL_TYPES,\@DATATYPES,$FILE_LAYOUT,$QA_ONLY,$UNIQUE_ID_TYPE,$CREATE_CLEAN_SLATE);
"""
pscript_file_layout = """'{}'=>{{
	    header_rows=>1,
	    layout=> {{
		state_id=> {},
		grade=> {},
		name=>{},
      	subject=> {},
		entity_type=>{},
		breakdown=>{},
		{}  # this entry and below governed by FileMap.data_types
	    		}}
	    }}
	    """


class Load:

	def __init__(self, info, name=None, qa_only=1):
		self.infiles = []
		self.state = info['state']
		self.ticket = info['ticket']
		self.data_types = info['data_types']
		self.year = info['year']
		self.source_id = info['source_id']
		self.load_type = info['load_type']
		self.qa_only = qa_only
		self.tables = {}
		self.push_path = info['push_path']
		self.name = '{} {} {}'.format(self.ticket,
									  self.state.upper(),
									  ','.join(self.data_types)) if name is None else name
		pscript_type_indicator = self.load_type if len(self.data_types) > 1 else list(self.data_types.keys())[0]
		self.perl_script_fname = '{}_{}_{}_{}.pl'.format(self.ticket,
								self.state, 
							normalize(pscript_type_indicator),
								  self.year)
		self.perl_script = None
		self.qconfig_name = 'queue.config.{}.{}.test.1.txt'.format(self.state, self.year)
		norm_name = normalize(self.name)
		self.update_qscript_name = '{}.update_queue_file.sh'.format(norm_name)
		self.QA_file = norm_name + '.QA_file.txt' 
		self.QA_table = None # until we get the QA file back.
		self.unmatched = None # until we get qconfig
		

	def __getitem__(self, key):
		if isinstance(key, str):
			return self.tables[key]
		else:
			raise IndexError('Item selection selects from LoadTables')

	def __iter__(self):
		return iter(self.tables.values())

	def add_tables(self, infiles, skip_columns_called=False, breakout=False, name_change=True):
		# make a hash with keys as output filenames and values as LoadTable objects derived from the input filenames
		add_ins = {outgen(out, self.ticket) if name_change else out: LoadTable(out, self, skip_columns=skip_columns_called, breakout_set=breakout)
						for out in infiles}
		for entry in add_ins:
			add_ins[entry].load = self
		self.tables.update(add_ins)
		self.infiles += infiles


	def get_rowcounts(self):
		breakdown = {outgen(tb.infile_name, self.ticket):len(tb.table) for tb in self}
		breakdown['TOTAL'] = sum(breakdown.values())
		return breakdown

	def valcount(self, targ):
		breakdown = {}
		total = 0
		print('Instances of {} Found:'.format(targ))
		for tb in self:
			count = tb.valcount(targ, tcol='all', show_inds = False)
			breakdown[tb.infile_name] = count
			print('{}: {}'.format(tb.infile_name, count))
			total += count
		print('\nTotal instances: {}'.format(total))
		breakdown['all'] = total
		return breakdown


	def group_x_by_y(self, xmap, ymap):
		'''a "group by" method for load objects,
		all tables in load must be mapped for this to work'''
		all_groupings = {}
		for tb in self:
			xind = getattr(tb.fmap, xmap)
			yind = getattr(tb.fmap, ymap)
			if type(yind) == str:
				print('Load Table File: {} has {} mapped as a constant value: {}, skipping this file in groupings.'.format(tb.infile_name, ymap, yind))
				continue
				# print to stdout warning that file will be skipped in groupings due to constant value in mapping
			tb_groupings = tb.group_x_by_y(xind, yind)
			for xkey in tb_groupings:
				if xkey in all_groupings.keys():
					all_groupings[xkey] = set(list(all_groupings[xkey])+list(tb_groupings[xkey]))
				else:
					all_groupings[xkey] = tb_groupings[xkey]
		return all_groupings




	def QA(self):
		for tb in self.tables:
			self.tables[tb].QA()

	def build_perl_script(self, unique_id='state_id', multiple_data_types=True):
		# build file layout hashes
		maparam = 'hash' if multiple_data_types else 'array'
		file_maps = ',\n'.join([fmap.perl_map() for fmap in [self.tables[tb].fmap for tb in self.tables]])
		# build full mappin' perl script
		data_type_as_string = ','.join(["'{}'".format(dt) for dt in self.data_types])
		layout_argv = [self.state, self.year, self.source_id, self.name, unique_id, data_type_as_string]
		#layout_argv = ["'{}'".format(i) if not str(i).isnumeric() else i for i in layout_argv]
		layout_argv.append(file_maps)
		layout_argv.append(self.qa_only)
		#print(layout_argv)
		self.perl_script = perl_script_outer_frame.format(*layout_argv)
		with open(self.perl_script_fname, 'w') as script:
			script.write(self.perl_script)
		print('perl script mapper saved to {}'.format(self.perl_script_fname))

	def push_tables(self, tag='', usr='ibicket', svr='datadev'):
		if not os.path.exists('load'):
			os.mkdir('load')
		for fname in self.tables:
			tb = self.tables[fname]
			self.tofile('./load/'+tag+fname)
		push(dest=self.push_path, infile='load/*', usr=usr, svr=svr)
		
		# 	tb.push_table(self.push_path, fname, tag=tag)
	

	def push_perl_script(self, server='datadev', user=uname, execute=False):
		if self.perl_script_fname is not None:
			push(infile=self.perl_script_fname,dest=self.push_path,svr=server,usr=uname)
			if execute:
				excommand = 'ssh -t {} "cd /{} && ./{}"'.format(server, self.push_path, self.perl_script_fname)
				os.system(excommand)
				#qa, queue, bash = self.pull_QA_queue()
				return
		else:
			print('Perl script mapper has not been built yet. Run Load method build_perl_script... then try the push method again')

	def build_queue_config_struct(self, queue_file_delimiter=':queue_modifier:'):

		qname = self.qconfig_name
		self.unmatched = read_qconfig(qname)
		return qstruct

	def test(self):
		print('{}: {} load consisting of {} files'.format(self.ticket, self.load_type, len(self.tables)))
		print('Load data type(s): {}'.format(', '.join(self.data_types)))
		for outfile in self.tables:
			tb = self.tables[outfile]
			print('\n\t{}\n'.format(outfile))
			tb.test()

	def tofile(self, tag='', fpath=None):
		for table in self.tables:
			tb = self.tables[table]
			tbname = tag+table
			print(fpath)
			tb.tofile()


	def get_test_results(self, mode='scp'):
		'''pulls QA and update queue config file'''
		if mode == 'scp':

			loc = self.push_path
			did_pull_qa = pull(loc+self.QA_file)
			if did_pull_qa:
				qa = QATable(self) # This declares the QATable object for the load
				self.QA_table = qa
			did_pull_queue = pull(loc+self.qconfig_name)
			if did_pull_queue:
				self.unmatched = self.build_queue_config_struct() # declares load.unmatched
			did_pull_bash_updater = pull(bash_updater_fname)
			if did_pull_bash_updater:
				self.update_queue_script = bash_updater_fname
		elif mode =='local':
			#self.QA_table = QATable(self)
			self.unmatched = self.build_queue_config_struct()
			#self.display_unmatched()

	def display_unmatched(self, print_queries=False):
		if self.unmatched is None:
			print('queue config file has not yet been built. Make sure it has been pulled to your working directory and build_queue_config_struct method has been run')	 
		else:
			if print_queries:
				qframe = "SELECT name, state_id, city, county, nces_code, type FROM _{}.{} WHERE NAME LIKE '%{}%'"
				query_fname = '{}_check_qconfig_queries.sql'.format(self.ticket)
				query_file = open(query_fname, 'w')
			# separate schools from districts:
			schools = [school for school in self.unmatched if school['school_or_district'].strip()=='school']
			districts = [dist for dist in self.unmatched if dist['school_or_district'] == 'district']
			unmtchd = {'school':schools, 'district':districts}
			queries={}
			# display results
			for etype in unmtchd:
				print('State IDs for the following {} {}s were not recognized by our database:\n'.format(len(unmtchd[etype]), etype))
				for entity in unmtchd[etype]:
					print('\t* | {} | State ID: {}'.format(entity['name'], entity['state_id']))
					if print_queries:
						query = qframe.format(self.state, etype, entity['name'])+';\n'
						query_file.write(query)
			if print_queries:
				query_file.close()

def sum_dicts(argv):
	output = {}
	for m in argv:
		output.update(m)
	return output


class FileMap:

	'''Each LoadTable object has an attribute fmap, that is an instance of this class,
	which maps the columns of a data file for the census load pipeline'''

	def __init__(self,fname,column_map, table):
		self.fname = fname
		self.data_types = column_map['data_types']
		self.table = table # LoadTable object of the mapping
		self.load = self.table.load # load of which this is a part
		self.load_col_names = ['state_id', 'entity_name', 'subject', 'grade', 'entity_type', 'breakdown']
		is_wide = False
		for col_val in self.load_col_names: # apply the attributes present in the keys of column_values param
			try:
				cval = column_map[col_val]
			except KeyError as e:
				print('column_map for {} is missing the key: {}'.format(self.fname, e))
				continue
			# if one of our target fields in the file is in wide format:
			if isinstance(cval, dict): # get the column indexes in the file, and add them to (or initialize if this is the first one)
				cval = self.table.hash_columns(cval) # the array of target indices
				if not is_wide:
					is_wide = True
					target_hash = cval
				else: # this is not the first wide-format field that has been encountered
					target_hash.update(cval) # add that hash to the output ^^^ does this actually happen ever?
			setattr(self, col_val, cval)
		if is_wide:
			self.target = list(target_hash.values())
			self.target_column_count = len(self.target)
		else:
			self.target = column_map['target']
			self.target_column_count = 1

		


	def perl_map(self):
		fmap_argv = [self.state_id, self.grade, self.entity_name, 
			self.subject, self.entity_type, self.breakdown]
		# wrap any strings in quotes
		pmap_args = ["'{}'".format(i) if type(i)==str and not (str(i).isnumeric()) else i for i in fmap_argv]
		# translate the syntax of any hash tables to perl
		for ind, val in enumerate(pmap_args):
			# for wide fields (represented as hashes) 
			if isinstance(val, dict):
				# wrap any integer values as arrays of length 1
				dct_arg = pmap_args[ind]
				for ckey in dct_arg:
					if isinstance(dct_arg[ckey], int):
						dct_arg[ckey] = [dct_arg[ckey]]
				dct_arg = to_perl_hash(dct_arg)
				pmap_args[ind] = dct_arg
		# now add the data_types hash formatted for perl to the end of pmap_argv
		pmap_args.append(to_perl_hash(self.data_types)[1:-1])

		# if render_data_types_as == 'array':
		# 	datypes = []
		# 	for dtype in self.data_types.values():
		# 		datypes += list(dtype.values())
		# 	datypes = str(datypes)
		# 	pmap_args.append(datypes)
		# elif render_data_types_as == 'hash':
		# 	print(self.data_types)
		# 	datypes = self.data_types
		# 	datypes = to_perl_hash(datypes)
		# 	pmap_args.append(datypes[1:-1])
		# else:
		# 	raise ValueError("Not sure what {} means as far as rendering data types that way goes...".format(render_data_types_as))
		#print(datypes)
		return pscript_file_layout.format(self.fname, *pmap_args)


	def clone(self, new_fname, alter):
		'''A method returning an identical copy
		of the FileMap object, save whatever
		alterations are indicated
		:param: new_fname; pretty self explanatory
		"param: alter: a hash with keys being a subset
		of the keys in column_map; these are the'''
		mapat = (v for v in dir(self) if not v.startswith('__') and not hasattr('__call__', v))
		new_cmap = {ckey: alter[ckey] if ckey in alter.keys() else getattr(self, ckey)}
		clone_obj = FileMap(new_fname, new_cmap)
		return clone_obj


	def alter(self, alter):
		'''Alter the attributes of a filemap object with a hash: keys=attribute names
		Values = New values'''
		#mapat = (v for v in dir(self) if not v.startswith('__') and not hasattr('__call__', v))
		new_cmap = {ckey: alter[ckey] if ckey in alter.keys() else getattr(self, ckey)
		for ckey in self.load_col_names}
		for attr in new_cmap:
			setattr(self, attr, new_cmap[attr])
			

class GsTable:
	
	def __init__(self, indat, load, skip_columns=False, from_array=False,headed=True):
		self.headed=headed
		if from_array:
			self.infile_name = '' # this is no longer needed; subtable class takes care of it
		else:
			self.infile_name = indat
		self.table = make_table(self.infile_name, skip_columns=skip_columns)
		self.original = self.table # should this go in load table?
		self.headers = self.table.pop(0) if headed else None
		self.headers = [h.strip() for h in self.headers] if headed else None
		self.og_headers = self.headers
		self.load = load
		self.colhash = {col: ind for ind, col in enumerate(self.headers)}
		self.normalize_headers()
		
		
	def __getitem__(self, key):
		if isinstance(key, slice):
			return [self.table[i] for i in range(key.start, key.stop, key.step)]
		elif isinstance(key, str):
			return [row[self.colhash[key]] for row in self.table]
		return self.table[key]

	
	def __iter__(self):
		return iter(self.table)

	def index(self, column_name):
		return self.colhash[column_name]


	def group_x_by_y(self, xcol, ycol): # this hsould be a method of GSTable; not LoadTable
		'''returns a hash from a unique xcol; to each unique ycol value paired with that xcol'''
		xcol = self.colhash[xcol] if not isinstance(xcol, int) else xcol
		ycol = self.colhash[ycol] if not isinstance(ycol, int) else ycol
		groupings = {unx:set() for unx in self.value_set(xcol)}
		for ind, row in enumerate(self):
			cur_xval = row[xcol]
			groupings[cur_xval].add(row[ycol])
		return groupings


	def normalize_headers(self):
		self.headers = [normalize(head) for head in self.headers]
		self.colhash = {normalize(ckey):self.colhash[ckey] for ckey in self.colhash}

	def tofile(self, tag='', fpath=None, delim='\t', encode_as='utf-8'):
		if fpath is None:
			fpath = self.outfile_name
		else:
			fpath = fpath + '/' + self.outfile_name
		towork = self.table
		name = tag + self.outfile_name  #.split('/')[-1]
		with open(name, 'w', encoding=encode_as) as file:
			print('\n writing updated table from {} to file...\n'.format(name))
			file.writelines(delim.join(self.headers)+'\n')
			for row in towork:
				row = delim.join(str(val) for val in row)
				file.writelines(row+'\n')
		print('file saved to {}'.format(name))

	def intercomp(self, other, compare_column):
		'''what values in the compare_column of self are not in that of other?
		So, ideally call the method from the GsTable that is thought to be the
		superset of the other'''
		self_set = set(self[compare_column])
		other_set = set(other[compare_column])
		return [val for val in self_set if val not in other_set]

	def valcount(self, targ, tcol, printout=True, show_inds=True):
		# TODO: make this respond to regex
		count = 0
		rel_rows = []
		if isinstance(tcol, str) and tcol != 'all':
			tcol = self.colhash[tcol]
			column_of_interest = [v[tcol] for v in self.table]
		if type(targ) in (list, tuple):
			ishit = lambda v: True if v in targ else False
		else:
			ishit = lambda v: True if v == targ else False
		if isinstance(tcol, str): # ie) if all columns are specified
			for idx in range(len(self.headers)):
				column_of_interest = [v[idx] for v in self.table]
				for ind, val in enumerate(column_of_interest):
					if ishit(val):
						count += 1
					if show_inds:
						rel_rows.append(str(ind))
		else:
			for ind, val in enumerate(column_of_interest):
				if ishit(val):
					count += 1
					if show_inds:
						rel_rows.append(str(ind))
		if printout:
			if targ != '':
				print('\n{} instance of {} found in column {} of {}\n'.format(count,targ,tcol, self.infile_name))
			else:
				print('\n{} empty values found in column {} of {}\n'.format(count, tcol, self.infile_name))
			rate = round(count/len(self.table), 3)*100
			print('Value(s) {} make up {} \%\ of the values in column {}\n\n'.format(targ, rate, tcol))
			if show_inds and count > 0:
				print('{} found at the following rows:\n'.format(targ)+', '.join(rel_rows))
		return count

	def count_by_condition(self, condition):
		'''given a condition (function object returning boolean) as argument
		this method prints a report of how many rows are in the table for which
		the condition is true.'''
		count = 0
		for row in self:
			if condition(row):
				count += 1
		return count

	

	def valcount_if(self, tcol, condition):
		if isinstance(tcol, str):
			tcol = self.colhash[tcol]
		count = 0
		tab = self.table
		column_of_interest = [v[tcol] for v in tab]
		for val in column_of_interest:
			if condition(val):
				count += 1
		print(count)

	def restore(self):
		self.table = self.original
		self.headers = self.og_headers

	def test(self, select=False, condition=None,verbose=False):
		'''now tell me about this table...
		param: select -> list of integers corresponding to row numbers to print'''
		tab = self.table
		if select:
			verbose = True
		if verbose:	
			for row in tab:
				if select:
					drow = [row[i] for i in range(len(row)) if i in select]
				else:
					drow = row
				if condition is not None:
					if not condition(row):
						continue
				print(drow)
		rowcount = len(tab)
		colcount = len(self.headers)
		print('table with {} rows and {} columns'.format(rowcount,
			colcount))
		if self.headers is not None:
			print('headed by:\n')
			for head in self.headers:
				print(head)
		print('\n\n')
		if self.ismapped:
			print('Entity Types:\n')
			ets = self.value_set(self.fmap.entity_type)
			for et in ets:
				print(et)
			print('\n')
			print('Breakdowns:')
			bks = self.value_set(self.fmap.breakdown)
			for bk in bks:
				print(bk)
			sids = self.value_set(self.fmap.state_id)
			sid_lens = set([len(i) for i in sids])
			if len(sid_lens) != 1:
				print("Inconsistent lenghts among state ids in {}: is this ok?\n\n".format(self.infile_name))

	def value_set(self, col):
		'''return the set of all values occurring in a given column passed
		as parameter (string: column's header name)'''
		if isinstance(col, str):
			col = self.colhash[col]
		return sorted(list(set([row[col] for row in self.table])))
	

	
class QATable(GsTable):
	def __init__(self, infile_name=None,load='QA', headed=True):

		''':params 
			- qa_fname; filename for QA file
			- load_tables; list of LoadTable objects corresponding to this load
			- queue_config;  file NAME for queue config file pulled from dir in public drive.
			- infile_name; for QATable class, this should be qa file pulled from public
					'''

		self.load = load
		self.infile_name = self.load.QA_file if infile_name is None else infile_name
		GsTable.__init__(self, indat=self.infile_name,load=load, headed=True)
		
		

		# should be array of LoadTable objects
		# self.queue_config_fname = qconfig_fname
		# self.unrec = build_queue_config_struct(qconfig_fname)

	
	def get_dupes(self, bycol):
		if not isinstance(bycol, tuple):
			check_ind = self.colhash[bycol] if isinstance(bycol, str) else bycol
			check_row = [row[check_ind] for row in self.table]
		else:
			check_ind = tuple([self.colhash[head] if isinstance(head, str) else head for head in check_ind])
			check_row = [tuple([row[idx] for idx in check_ind]) for row in self.table]
		dupes = {}
		if len(set(check_row)) != len(check_row):
			# no need to iterate over rows if there are as many distinct values as values in general
			for row_num, idx in enumerate(check_row):
				if idx not in dupes.keys():
					dupes[idx] = [row_num]
				else:
					dupes[idx].append(row_num)
					print('Duplicate value found for {}\n'.format(', '.join(list(idx))))
			dupes = {dkey:dupes[dkey] for dkey in dupes if len(dupes[dkey]) > 1}
			for dupe in dupes:
				print('{} found in rows {}'.format(dupe, ', '.join(dupes[dupe])))
		else:
			print('No dupes in this one, at least not by column(s) {}'.format(bycol))
		return dupes

	def compare(self, to, bycol, verbose=True):
		'''param: to should be QATable Object (or LoadTable?)
		param: bycol: HASH, keys are headers (or column indices) inself, values in compare table'''
		# generate lambda function to get cells to compare (as tuple) for each row
		self_ind = tuple([self.colhash[head] if isinstance(head, str) else head for head in bycol])
		check_self = lambda row: tuple([row[idx] for idx in self_ind])
		other_ind = tuple([self.colhash[bycol[head]] if isinstance(bycol[head], str) else bycol[head] for head in bycol])
		check_other = lambda row: tuple([row[idx] for idx in other_ind])
		max_range = range(max((len(self.table), len(to.table))))
		self_compare = []
		other_compare = []
		for ridx in max_range:
			# get row in both tables if the tables are that long.
			self_row = self.table[ridx] if ridx < len(self.table) else None
			comp_row = to.table[ridx] if ridx < len(to.table) else None
			if self_row not in self_compare:
				self_compare.append(self_row)
			if comp_row in other_compare:
				other_compare.append(comp_row)

		self_not_other = [i for i in self_compare if i not in other_compare]
		other_not_self = [i for i in other_compare if i not in self_compare]
		if verbose:
			print('{} rows in {} not in {}'.format(len(self_not_other), self.infile_name, to.infile_name))
			print('{} rows in {} not in {}'.format(len(other_not_self), to.infile_name, self.infile_name))
		return self_not_other, other_not_self








	def QA(self, check_column=None):
		
		rowcount = sum([len(table) for table in self.load_tables])
		qa_count = len(self)
		loaded = None
		if len(rowcount) != len(qa_count):
			print('{} has {} data rows\nQA file has {} data rows'.format(loaded,rowcount_self,rowcount_qa))
		else:
			print('\nData gods be praised!!!\nQA file length matches!!\n')
		if len(self.load_tables) == 1:
			compare = self.load_tables[0]
			for ckey in compare.colhash:
				vset = compare.value_set(ckey)
				print('{} column has the following distinct values:'.format(ckey))
				print(vset)
				print('\n\n')
		else:
			for table in self.load_tables:
				continue
				
		
		# split queue file by queue delimiter and by new line;	
		
		new_count = len(self.unrec)

		if not new_count == 0:
			print('\n{} entries from input not recognized in GreatSchools database.\n'.format(new_count))
			schools = [(entry['name'], entry['state_id']) for entry in self.unrec 
			if entry['school_or_district'].strip() == 'school']
			districts = [(entry['name'], entry['state_id']) for entry in self.unrec 
			if entry['school_or_district'].strip() == 'district']
			print('Unrecognized Schools:')
			if len(schools) is not None:
				for school in schools:
					print('{}, state_id: {}\n'.format(school[0], school[1]))
			if len(districts) is not None:
				print('\nUnrecognized Districts:')
				for district in districts:
					print('{}, state_id: {}\n'.format(district[0], district[1]))

			return False
		else:
			print('No unrecognized entities here according to the queue file...')
			return True




		
class LoadTable(GsTable):

	def __init__(self, infile_name, load, skip_columns=False, headed=True, breakout_set=False, from_array=False):
		### PUT A FUNCTION HERE THAT CONCATS TO GIVE BREAKOUT COLUMNS UNIQUE COLHASH KEYS###
		GsTable.__init__(self, infile_name, load=load, skip_columns=skip_columns)
		# concatenate breakout identifier with each repeated header that comprises breakout column
		# these should take the form of a subtable
		self.normalize_headers()
		if breakout_set:
			# get headers and breakout values
			self.breakouts = self.headers
			self.headers = self.table.pop(0)
			self.normalize_headers()
			# get the indices of the breakout tags
			bk_tags = [ind for ind, val in enumerate(self.breakouts) if val != '']
			# check the header value they occur directly above
			bk_headcheck = [self.headers[ind] for ind in bk_tags]
			# check to see that they are all the same
			if len(set(bk_headcheck)) == 1:
				bk_above = next(iter(bk_headcheck))
				# also make sure that they are in fact in the breakout set
				assert bk_above in breakout_set
			else:
				raise ValueError('Check your file; breakout tags do not all correspond to the same header tag')
		# get the breakout set's index for the header value occurring below breakout tags
			bk_ind = breakout_set.index(bk_above)
		# now concatenate breakout tags to each column associated with it
			# get the left and right span of the breakout set to create a range relative to bk_ind
			bk_span = len(breakout_set)
			for idx in bk_tags:
				left_span = idx - bk_ind
				right_span = left_span + bk_span
				cur_range = range(left_span, right_span)
				# concatenate breakout tag to each breakout set, creating a unique single row of headers.
				for i in cur_range:
					self.headers[i] = self.breakouts[idx] + '_' + self.headers[i]
			self.colhash_refresh()
		self.fmap = None
		self.ismapped = False
		self.subtables = []
		self.outfile_name = outgen(self.infile_name, self.load.ticket)
		self.clear_whitespace()
		
	def map(self, fmap):
		'''NOTE: data types should be a key in fmap, indicating which columns are for which data type
		In this way, we can take the intersection of another column hash and a data types hash
		to get the '''
		self.fmap = FileMap(outgen(self.infile_name, self.load.ticket), fmap, self)
		self.ismapped = True

	def from_array(self, array, filetype='.txt', tag=''):
		'''initializes LoadTable object from a 2D array rather than from a file'''
		output = LoadTable(self.infile_name[:-4]+tag+filetype, self.load, skip_columns=None, from_array=True)
		output.fmap = self.fmap
		return output	

	def head_rename(self, old, new):
		# create a new hash entry and 
		old_header_index = self.colhash[old]
		self.colhash[new] = old_header_index
		# delete the old
		del self.colhash[old]
		# now the same for the header list
		del self.headers[old_header_index]
		self.headers.insert(old_header_index,new)
		# re-sort (is this necessary?)
		self.headers = sorted(self.headers, key=lambda x: self.colhash[x])

	def clean_reduced_rows(self):
		indices = []
		iscompd = []
		# iterate over subtables, and remove rows from table
		# if they are included in a subtable that has been computed
		for sbt in self.subtables:
			if sbt.computed:
				indices += sbt.inds
				iscompd.append(sbt)			
		self.table = [row for (ind, row) in enumerate(self.table) if ind not in indices]

	def dupe_scan(self, by_column):
		dupes = {}
		ncols = isinstance(by_column, tuple)
		if ncols: # get duplicates by multiple columns
			tp_idx = []
			for col_ind in by_column:
				if isinstance(col_ind, str):
					col_ind = self.colhash[col_ind]
				tp_idx.append(col_ind)
			tp_idx = tuple(tp_idx)
			names = ' x '.join(list(by_column))
			by_column = tp_idx
		for ind, row in enumerate(self): 
			value = row[self.colhash[by_column]] if isinstance(by_column, str) else row[by_column] if isinstance(by_column, int) else tupind(row, by_column)
			if value not in dupes.keys():
				dupes[value] = [ind]
			else:
				dupes[value].append(ind)
		dupes = {dkey:dupes[dkey] for dkey in dupes if len(dupes[dkey]) > 1}
		if len(dupes) != 0:
			print('{} n-plicate {} values found in {}:\n'.format(len(dupes), by_column, self.infile_name))
			#print('\n')
			for dupe_val in dupes:
				print(dupe_val)
				dupe_bank = dupes[dupe_val]
				for dupe in dupe_bank:
					#print('\t'+'| '.join(self.table[dupe]))
					print(dupe)
				print('\n\n')
		else:
			print('\nNo duplicate {} values found in {}\n'.format(names, self.infile_name))
		return dupes





	def hash_columns(self, inhash):
		output = {}
		#{ckey: self.colhash[inhash[ckey]] for ckey in inhash}
		for ckey in inhash:
			if isinstance(inhash[ckey], list):
				output[ckey] = [self.colhash[i] for i in inhash[ckey]]
			else:
				output[ckey] = inhash[ckey]
		return output

	def split_by(self, condition,tag_true="I", tag_false="II", by_column=False, to_file=False):
		'''This method returns 2 tables split by a condition.

		Note: this passing of parameters for identity conditions is
		UGLY!! make this better...''' 
		print('splitting table {}'.format(self.infile_name))
		#print('separating files by condition {} == {}...'.format(tcol, tval))
		#tcol = self.colhash[tcol]
		table_a = [row for row in self.table if condition(row)]
		table_b = [row for row in self.table if not condition(row)]
		print('copying...')
		tb1 = dpc(self)
		tb1.table = table_a
		tb2 = dpc(self)
		tb2.table = table_b
		# delete original table from load
		tb1_name = outgen(self.infile_name, self.load.ticket)[:-4]+'_{}.txt'.format(tag_true)
		tb1.infile_name = tb1_name
		tb2_name = outgen(self.infile_name, self.load.ticket)[:-4]+'_{}.txt'.format(tag_false)
		tb2.infile_name = tb2_name
		self.load.tables[tb1_name] = tb1
		self.load.tables[tb2_name] = tb2
		if to_file:
			for name_tb in ((tb1_name, tb1),(tb2_name, tb2)):
				name, tb = name_tb[0], name_tb[1]
				tb.tofile(fpath=name)
		return tb1, tb2
		
	
	def select_by(self, tcol=None, tval=None, condition=None, return_subtable=True):
		if tcol is not None:
			tcol = self.colhash[tcol] if not isinstance(tcol, int) else tcol
		# enumerate the indices of each row and [if condition != None] select a subset from which to group records
		if tval is not None:
			if condition is not None:
				condition = lambda drow: True if condition(row) and row[tcol] == tval else False
			else:
				condition = lambda drow: True if row[tcol] == tval else False
		select = [(ind, row) for ind, row in enumerate(self.table) if condition(row)]
		if return_subtable:
			if len(select) ==0:
				return None
			else:
				return SubTable(select, self)
		else:
			return select

	



	def max_length(self, column_name):
		'''return the maximum length of a column'''
		return max([len(i) for i in tb[column_name]])

	def derive_new_column(self, columns, procedure, newcol_name):
		'''
				-A method to derive a new column from values in existing columns-
		columns should be a list of numerical indices
		procedure should be a function object that, given a GsTable object and a list of header names
		returns a new column in the form of a list
		'''
		#colinds = [self.colhash[col] if isinstance(col, str) else col for col in columns]
		tab = self.table
		self.headers.append(newcol_name)
		# update column hash for column slicing.
		self.colhash[newcol_name] = len(self.headers)-1
		ccount = len(self.headers)
		new_col = procedure(self, columns) # procedure should take whole list of srows
		for ind, val in enumerate(new_col):
			tab[ind].append(val) # append each new value to the appropriate row
		self.table = tab

	def const_column(self, header, cval=''):
		'''add a column to the table with a constant value
		if not specified, defaults to empty string'''
		tab = self.table
		self.headers.append(header)
		self.colhash[header] = len(self.headers)-1
		for row in tab:
			row.append(cval)
		self.table=tab

	
	def clear_whitespace(self, column='all'):
		if column == 'all':
			for head in self.headers:
				head_ind = self.colhash[head]
				for row in self:
					row[head_ind] = row[head_ind].strip()
		else:
			head_ind = self.colhash[column]
			for row in self:
				row[head_ind] = row[head_ind].strip()

	

	def valtweak(self, tcol, tweaker, condition=None, verbose=False, headed=True):
		'''Do something to column[tcol] in each row if condition(s) are
		met
		NOTE: tweaker parameter should take row as argument and return cell value
		NOTE: condition should take row as arg and return boolean'''
		if isinstance(tcol, str) and tcol != 'all':
			tcol = self.colhash[tcol]
		tab = self.table
		if condition is not None:
			# convert condition to an identity constraint function if it is not
			# already a function object
			if type(condition) == str:
				# todo: make this handle numerical data
				target = condition.strip() 
				condition = lambda r: True if r[tcol].strip() == target else False
			elif type(condition) in (list, tuple):
				target = tuple([i.strip() for i in condition])
				condition = lambda r: True if r[tcol].strip() in target else False
		count = 0
		for row in tab:
			bef = row[tcol]
			#if condition is not None: # test the condition
			ismet = condition(row) if condition is not None else True
			if not ismet:
				#print('CONDITION WAS MET')
				continue
			if verbose:
				print("CONDITION WAS MET")
			if tcol =='all':
				row = [tweaker(head) for head in row]
			else:
				row[tcol] = tweaker(row)
			if row[tcol] == bef:
				count += 1
		tcol = self.headers[tcol] if isinstance(tcol, int) else tcol
		print("\n{} cells in table {} at column {} affected by {}\n".format(count, self.infile_name, tcol, tweaker.__name__))
		return count
		
	def remove_rows_if(self, condition, tcol=None, testit=False, verbose=True, store=False):
	
		''':param condition, function object returning
		a boolean value if a condition is met.
			:param table; table to return a cleaned version of
		:return; table cleared of all rows that satisfy the condition
			parameter'''
		count = 0
		removed = []
		tab = self.table
		output = []
		for row in tab:
			if tcol is None:
				should_remove = condition(row)
			else:
				if isinstance(tcol, str):
					tcol = self.colhash[tcol]
				should_remove = condition(row[tcol])
			if should_remove:
				if testit:
					print('REMOVAL HAPPENED!: trigger_value: {}'.format(row[tcol]))
				count += 1
				if store:
					removed.append(row)
				continue
			output.append(row)
		self.table = output
		if verbose:
			print('\n\n{} rows removed from {} on account of {}\n\n'.format(count, self.infile_name, condition.__name__))
		return count if not store else removed
	
	def check_rows_if(self, condition):
		matches = [(i, row) for (i, row) in enumerate(self.table) if condition(row)]
		#print('C: | {}'.format(self.headers))
		# for i in range(len(matches)):
		# 	print('{}: | {}'.format(i, ' | '.join(matches[i])))
		return matches




	def change_summary(self):
		row_change = len(self.table)-len(self.original)
		col_change = len(self.headers)-len(self.og_headers)
		if row_change < 0:
			verb = 'subtracted'
			prp = 'from'
			row_change = 0-row_change
		else:
			verb = 'added'
			prp = 'to'
		print('{} {} rows {} input file'.format(verb, row_change, prp))
		if col_change < 0:
			verb = 'subtracted'
			prp = 'from'
			col_change = 0-col_change
		else:
			verb = 'added'
			prp = 'to'
		print('{} {} columns {} input file'.format(verb, col_change, prp))



	def vert_split(self, columns, typehead, valhead):
		
		'''makes unique rows for each of a set
		of columns with the original 
		columns distinctive factor recoded
		as the value of another column; ie) wide to long format.
		param: 
		columns- a list of tuples of the form (old_column, new_id_value)
		these should correspond to a number of column heads (index 0)
		and their corresponding indicator values (index 1) in the new
		long format column 
		typehead: a string indicating the header that will display the 
		type of value.
		valhead: a string, should indicate the column that has the value'''

		tab = []
		split_count = len(columns)
		# get column indices to remove
		old_col_heads = [c[0] for c in columns]
		new_col_vals = [c[1] for c in columns]
		old_col_inds = [self.colhash[head] for head in old_col_heads]
		lftag_mapper = list(zip(old_col_inds, new_col_vals))
		# add new columns to headers and remove the old
		self.headers = [head for head in self.headers if head not in old_col_heads]
		for new_head in (typehead, valhead): # append 
			self.headers.append(new_head)
		self.colhash_refresh()
		
		# indices for new columns with respect to filled row
		# DON'T use these to slice rows from self.table
		type_ind = self.colhash[typehead] # this may be redundant mapper[n][0]
		val_ind = self.colhash[valhead]
		
		# iterate over rows
		for row in self.table:
			# create template from old row
			skeleton = [row[i] for i in range(len(row)) if i not in old_col_inds]
			# create a new row with the new columns added to the table above
			#print(count)
			for ocind, lftag in lftag_mapper: # ocind is the old row's index for type
				new_row = [bone for bone in skeleton] # add 2 columns		
				new_row.insert(self.colhash[typehead], lftag) # insert type column value #TC subject
				#print(ocind, lftag)
				new_row.insert(val_ind, row[ocind])
				# add the row to the new longform table
				tab.append(new_row)
		# sort headers
		# is this still necessary?
		#self.headers = sorted(self.headers, key=lambda x: self.colhash[x])
		self.table = tab
		
	def mean(self, colname, printout=False):
		column = self[colname]
		if not all([s.isnumeric() for s in column]):
			print("Can't compute mean value for {}; not all values are numeric".format(colname))
			outval = None
		else:
			column = [float(val) for val in column]
			mean_val = sum(column)/len(column)
			if printout:
				print('\nmean value for {}: {}\n'.format(colname, str(mean_val)))
			outval = mean_val
		return outval




	def colhash_refresh(self):
		self.colhash = {head:ind for ind, head in enumerate(self.headers)}

	def x_then_y_is_z(self, xcol, xis, ycol, yis, shift_from=False):
		'''updates column y to value z if another column has value x 
		:param xcol should be the index of the triggering condition

		NOTE: if shift_from parameter is set to True, yis is inerpreted as a column
				header; the current row's value for which will be transferred to the target column'''
		xind = self.colhash[xcol] if isinstance(xcol, str) else xcol
		yind = self.colhash[ycol] if isinstance(ycol, str) else ycol
		tab = self.table
		for row in tab:
			if type(xis) == list:
				xis = [i.lower() for i in xis]
				trcond = (row[xind].lower() in xis)
			elif type(xis) == str:
				xis = xis.lower().strip()
				trcond = (row[xind].lower().strip() == xis.lower().strip())
			else:
				raise TypeError('xis parameter must be of type string or list, not {}'.format(type(xis)))
			if trcond:
				# set the value of the specified column to the specified value if the condition is met. 
				if shift_from:
					row[yind] = row[self.colhash[yis]]
				else:
					row[yind] = yis
		self.table=tab
	
	def change_to_x_if(self, xcol, trigger_val, outval):
		'''If a value == trigger_val, replace it with outval'''
		self.x_then_y_is_z(xcol, trigger_val, xcol, outval)

	

	def examine_column(self, column):
		if isinstance(column, str) and column not in self.headers:
			print('\nNo column called {}'.format(column))
		else:
			print('Column: {}\n\tContaining the following values;\n'.format(column))
			content = self.value_set(column)
			count_distinct = len(content)
			pad = '\t\t\t\t'
			for v in content:
				if v.strip() == '':
					print(pad+'EMPTY VALUE')
				else:
					print(pad+v)
			print('\t{} distinct values\n'.format(count_distinct))
			num_vals = [i for i in content if isfloat(i)]
			if len(num_vals) > 0:
				quant = [float(n) for n in num_vals]
				min_val = min(quant)
				max_val = max(quant)
				print('\nMininum Value: {}\nMaximum Value: {}'.format(min_val, max_val))

	def push_table(self, remote_path, fname, tag='', server='datadev', user=uname):
		if not os.path.exists('load'):
			os.mkdir('load')
		self.tofile('load/'+tag+fname)
		push(dest=remote_path,infile=fname,usr=user,svr=server)


	def QA(self):
		if self.fmap is None:
			raise ValueError('Load table has not yet been mapped')
		else:
			qa_log_fname = outgen(self.infile_name, self.load.ticket)[:-4]+'_'+'QA_log_file.txt'
			qa_log_file = open(qa_log_fname, 'w')
			qaf_header = '\n\t\tQA for Preprocessing Work On {}\n\n'.format(self.infile_name)
			qa_log_file.write(qaf_header)
			print(qaf_header)
			state_id_registry = {sid:[] for sid in set([row[self.fmap.state_id] for row in self.table])}
			invalid_sid_blanks = []
			grade_problems = 0
			brkdwn_problems = 0
			subj_problems = 0
			wegood = True
			print('\nchecking rows for invalid or duplicate values...\n')
			for ind, row in enumerate(self.table):
				state_id = row[self.fmap.state_id]
				# get grade validations
				if isinstance(self.fmap.grade, int):
					grade = row[self.fmap.grade]
					grade_check = valid_grade(grade)
					if not grade_check:
						grade_problems += 1
				else:
					grade_problems = "N/A"
				# census breakdown validations
				if isinstance(self.fmap.breakdown, int):
					brkdwn = row[self.fmap.breakdown]
					brkdwn_check = valid_breakdown(brkdwn)
					if not brkdwn_check:
						brkdwn_problems += 1
				else:
					brkdwn_problems = "N/A"
				# subject validations
				if isinstance(self.fmap.subject, int):
					subj = row[self.fmap.subject]
					subj_check = valid_subj(subj)
					if not subj_check:
						subj_problems += 1
				else:
					subj_problems = "N/A"
				if all([field=="N/A" for field in (grade_problems, brkdwn_problems, subj_problems)]):
					checks = [True, True, True]
					sawit = '\nBut grade, subject and breakdown were all values that cant be checked by the current procedure...'
				else:
					sawit = ''
					checks = [grade_check, subj_check, brkdwn_check]
				fields = (grade, subj, brkdwn)
				state_id_registry[state_id].append((ind, fields+(row[self.fmap.entity_type],)))
				# if any of the checks pass as false
				if not all(checks): # if anything is wrong with this row
					wegood = False
					
					row_report = 'ROW {}: |'.format(ind)
					to_report = [fields[i] for i in range(len(checks)) if checks[i]]
					to_report = row_report +' | '.join(to_report) + ' |\n'
					qa_log_file.write(to_report)
			# final report
			summary_header = '_'*25+'\n\nSummary:\n'
			print(summary_header)
			qa_log_file.write(summary_header)
			if not wegood:
				counts = [grade_problems, subj_problems, brkdwn_problems]
				val_report = "Number of invalid values for:\nGrade: {}\nSubject: {}\nBreakdown: {}\n\n".format(*counts)
				print(val_report)
				qa_log_file.write(val_report)
				
			else:
				good_message = '\n\t\t¯\_(ツ)_/¯\n\t\tRows Look good to me! {} \n\n'.format(sawit)
				print(good_message)
				qa_log_file.write(good_message)
			# address duplicate or blank state_ids
			blank_sid_list = state_id_registry['']
			del state_id_registry['']
			#state_id_registry = {sid:state_id_registry[sid] for sid in state_id_registry if len(state_id_registry[sid]) > 1 and sid != ''}
			print('\n\nchecking table for duplicate data points...\n\n')
			sid_dupes = get_duplicate_values(state_id_registry)
			sidupe_count = len(sid_dupes)
			sid_dupe_report ="{} duplicated state_id values found\n\n".format(sidupe_count)
			print(sid_dupe_report)
			qa_log_file.write(sid_dupe_report)
			if sidupe_count != 0:
				qa_log_file.write('Duplicate state ids found at the following row indices:\n\n')
				sid_dupes = [str(dupe)[1:-1]+'\n\n' for dupe in sid_dupes]
				qa_log_file.writelines(sid_dupes)
			# now address the blank state ids, ignoring those for state-level data
			if isinstance(self.fmap.entity_type, str):
					# if entity type is constant throughout the file
					if not self.fmap.entity_type.strip() == 'state': # and it's not all state-level
						blanks = blank_sid_list
					else:
						blanks = []
			else:
				blanks = [i[0] for i in blank_sid_list if self.table[i[0]][self.fmap.entity_type].strip() != 'state']
			blcount = len(blanks)
			blank_sid_summary = "\n\n{} blank values for state_id\n\n".format(blcount)
			print(blank_sid_summary)
			qa_log_file.write(blank_sid_summary)
			for bl in blanks:
				report_string = '\tRow #: {} | Entity Name {}\n'.format(bl, self.table[bl][self.fmap.entity_name])
				qa_log_file.write(report_string)

			qa_log_file.close()




class SubTable(LoadTable):
	'''A SubTable class is intended for situations in which n rows are to be reduced to 1 by some procedure
	It should therefore be an attribute of a LoadTable with an attribute indicating the original indices
	of its rows in the LoadTable. This can be used to easily delete the original rows from the table, replacing them with'''
	# NOTE: what goes in here should be the a list of tuples like (original_table_ind, [row0, row1...rown])
	def __init__(self, rows, tb, headed=False):
		#print(rows)
		if len(rows) == 0:
			raise ValueError("Come now, there's no reason to initialize a SubTable of 0 rows, that's just silly...")
		self.inds = sorted([i[0] for i in rows], reverse=True)
		self.table = [i[1] for i in rows]
		self.template_row = []
		#print(self.table[rows[0][0]])
		self.load_table = tb # should be loadTable object of which this table is a subset
		self.load_table.subtables.append(self)
		self.headers = self.load_table.headers
		self.colhash = self.load_table.colhash
		# build template row (common fields, with empty string placeholders where fields are not common)
		for column in self.headers:
			# fields common among all rows in subtable are passed to reduced output row
			if all([v == self[column][0] for v in self[column]]):
				self.template_row.append(self[column][0])
			else:
				self.template_row.append('')
		self.computed = False

	def assimilate_as(self,indicator, source_column, target_column):
		'''source_column: column in the subtable's template row that represents the target value of the subtable's row
		target_column: column where the source column's value should be transferred to: this is ideally empty in the template row
		indicator: either (tcol, tval) or lambda row: row[tcol] = tval'''
		# format parameters to generate new row
		if isinstance(indicator, tuple):
			tcol = indicator[0] if isinstance(indicator[0], int) else self.colhash[indicator[0]]
			tval = indicator[1]
			indicator = lambda row: [row[i] if i != tcol else tval for i in range(len(row))] # should be lambda function of row otherwise
		source_column = self.colhash[source_column] if isinstance(source_column, str) else source_column
		target_column = self.colhash[target_column] if isinstance(target_column, str) else target_column
		ccount = range(len(self.headers))
		# construct new row
		new_row = [self.template_row[i] if i != target_column else self.template_row[source_column] for i in ccount]
		new_row = indicator(new_row)
		self.load_table.table.append(new_row)




	def weighted_avg(self, weight_col, value_col):
		'''compute a weighted average of one or more columns with each row
		weighted by value of the specified weight_col'''
		weights = [num_norm(val) for val in self[weight_col]]
		weights = [float(val) for val in weights]
		denom = sum(weights)
		if isinstance(value_col, str):
			# floatify and calculate weighted values
				values = [float(num_norm(val)) for val in self[vcol]]
				values = [weights[ind]*val for ind, val in enumerate(values)]			
				# calculate mean
				num = sum(values)
				avg = round(num/denom, 3)
				out = {vcol:avg}
		elif type(value_col) in (list, tuple):
			out = {}
			for vcol in value_col:
				# floatify and calculate weighted values
				values = [float(num_norm(val)) for val in self[vcol]]
				values = [weights[ind]*val for ind, val in enumerate(values)]			
				# calculate mean
				num = sum(values)
				avg = round(num/denom, 3)
				out.update({vcol:avg})
		return out
		

	def table_reduce(self, reduced_value):
		'''given a hash of column headers mapped to a reduced value,
		return a single compressed row of the superset LoadTable object'''
		# compose reduced row from input values
		
		for col in reduced_value:
			red_val = reduced_value[col]
			trow_ind = self.colhash[col]
			self.template_row[trow_ind] = red_val # assign reduced value to 
		# remove the original rows and append
		#self.table_replace()
		self.computed = True  # indicate to the load table that it's good process this table's reduction
		return self.template_row

	def table_replace(self):
		'''THIS METHOD IS CURRENTLY DEPRECATED;
		TODO: right now, this method would have removed a number of rows and replaced them with 1
		the problem is that subtables keep track of the rows they are to remove by the index in the original
		load table object, so calling this function basically renders those indices bullshit'''
		# rebuild load table without the initial rows
		self.load_table.table = [row for ind, row in enumerate(self.load_table.table) if ind not in self.inds]
		# add derived row to table
		self.load_table.table.append(self.template_row)





