import matplotlib.pyplot as plt
import logging

def export_plot(ys, ylabel, filename):
	"""
	Export a plot in filename

	Args:
		ys: (list) of float / int to plot
		filename: (string) directory
	"""
	plt.figure()
	plt.plot(range(len(ys)), ys)
	plt.xlabel("Epoch")
	plt.ylabel(ylabel)
	plt.savefig(filename)
	plt.close()


def get_logger(filename):
	"""
	Return a logger instance to a file
	"""
	logger = logging.getLogger('logger')
	logger.setLevel(logging.DEBUG)
	logging.basicConfig(format='%(message)s', level=logging.DEBUG)
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
	logging.getLogger().addHandler(handler)
	return logger