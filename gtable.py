#!/usr/local/bin/python3

import re, csv
from utils import *

class Row:
    def __init__(self, dat, idx):
        self.dat = dat
        self.idx = idx # should be index from original table
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.dat[i] for i in range(key.start, key.stop, key.step)]
        elif isinstance(key, int):
            return self.dat[key]
        else:
            raise ValueError("can't select row index by {} instance".format(type(key)))
        

class GTable:
	
	def __init__(self, indat, skip_columns=False, from_array=False,headed=True):
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
        self.table = [Row(row, i) for i, row in enumerate(self.table)]
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


	def group_x_by_y(self, xcol, ycol): 
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
        
    
	def value_set(self, col):
		'''return the set of all values occurring in a given column passed
		as parameter (string: column's header name)'''
		if isinstance(col, str):
			col = self.colhash[col]
		return sorted(list(set([row[col] for row in self.table])))
        
	def select_by(self, tcol=None, tval=None, regex=None condition=None):
        if all(i is None for i in (tval, regex, condition)):
            raise ValueError('Please enter some criterion by which to select rows')
		if tcol is not None:
			tcol = self.colhash[tcol] if not isinstance(tcol, int) else tcol
		# enumerate the indices of each row and [if condition != None] select a subset from which to group records
		if tval is not None:
			if condition is not None:
				condition = lambda drow: True if condition(row) and row[tcol] == tval else False
			else:
				condition = lambda drow: True if row[tcol] == tval else False
		select = [(ind, row) for ind, row in enumerate(self.table) if condition(row)]
		return select
    
	def const_column(self, header, cval=''):
		'''add a column to the table with a constant value
		if not specified, defaults to empty string'''
		tab = self.table
		self.headers.append(header)
		self.colhash[header] = len(self.headers)-1
		for row in tab:
			row.append(cval)
		self.table=tab
        
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

class Selection(GTable):
    '''subclass of GTable meant to be returned by "select_by" and "group_by" methods'''
    def __init__(self, source):
        GTable.__init__()

if __name__ == '__main__':
    # test stuff
    testfile = ''
