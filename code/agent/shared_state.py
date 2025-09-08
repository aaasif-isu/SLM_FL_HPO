# agent/shared_state.py

from queue import Queue

# This queue will hold the results from the trainer.
# The format for items will be: (client_id, results_dict)
results_queue = Queue()