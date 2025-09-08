# agent/cpu_worker.py

from .shared_state import results_queue
from .workflow import create_cpu_graph

def background_cpu_work(client_states):
    hpo_graph = create_cpu_graph()

    while True:
        client_id, training_results = results_queue.get()

        # Acknowledge the task *before* checking if it's the stop signal.
        # This ensures the main thread's .join() call can complete.
        if client_id is None:
            results_queue.task_done() # Acknowledge the 'None' task
            break                 # Then exit the loop

        print(f"[CPU Worker]: Analyzing results for client {client_id}...")

        state_for_graph = {**client_states[client_id], **training_results}
        final_state_from_graph = hpo_graph.invoke(state_for_graph)

        client_states[client_id]['search_space'] = final_state_from_graph.get('search_space', {})
        client_states[client_id]['concrete_hps'] = final_state_from_graph.get('hps', {})
        client_states[client_id]['last_analysis'] = final_state_from_graph.get('last_analysis', {})

        print(f"[CPU Worker]: New HPs and search space for client {client_id} are ready.")
        
        # This call handles all the regular tasks.
        results_queue.task_done()