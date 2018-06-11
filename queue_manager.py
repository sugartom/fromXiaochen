import Queue 
from time import sleep 

class Q:
	def __init__(self, qsize):
		self.q = Queue.Queue(qsize)

	def write(self, v):
		keep_input = True
		while keep_input: 
			try:
				self.q.put_nowait(v)
			except Queue.Full:
				sleep(0.001)
				continue
			keep_input = False


	def force_write(self, v):
		keep_input = True
		while keep_input: 
			try:
				self.q.put_nowait(v)
			except Queue.Full:
				_ = self.read()
				continue
			keep_input = False


	def read(self):
		while True:
			try:
				d = self.q.get()
			except Queue.Empty:
				sleep(0.001)
				continue
			return d
