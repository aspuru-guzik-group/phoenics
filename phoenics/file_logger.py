#!/usr/bin/env python 

from watchdog.observers import Observer 
from watchdog.events    import PatternMatchingEventHandler

#======================================================================

class FileHandler(PatternMatchingEventHandler):

	PATTERN = ['*status']

	def __init__(self, event):
		PatternMatchingEventHandler.__init__(self)
		self.process_event = event

	def process(self, found_file):
		file_name = found_file.src_path
		self.process_event(file_name)

	def on_created(self, found_file):
		self.process(found_file)

#======================================================================

class FileLogger(object):

	def __init__(self, action, path = './'):
		self.path = path 
		self.event_handler = FileHandler(action)

	def start(self):
		self.observer = Observer()
		self.observer.schedule(self.event_handler, self.path, recursive = True)
		self.observer.start()

	def stop(self):
		self.observer.stop()

#======================================================================
