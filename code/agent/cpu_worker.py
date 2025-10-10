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
            results_queue.task_done()  # Acknowledge the 'None' task
            print(" All tasks complete. System shutting down.")
            break

        # Phase/round are now provided by trainer context (Step 1)
        phase = training_results.get("phase")
        round_idx = training_results.get("round_idx")
        total_rounds = training_results.get("total_rounds")

        # Backfill from global_epoch/global_epochs if needed
        if round_idx is None and "global_epoch" in training_results and "global_epochs" in training_results:
            training_results["round_idx"] = int(training_results["global_epoch"]) + 1
            training_results["total_rounds"] = int(training_results["global_epochs"])
            round_idx = training_results["round_idx"]
            total_rounds = training_results["total_rounds"]


        print(f"[CPU Worker]: Analyzing results for client {client_id} "
              f"[phase={phase}, round={round_idx}/{total_rounds}]...")

        # Merge immutable client state with latest training results for the graph
        state_for_graph = {**client_states[client_id], **training_results}

        # Run ANALYZE -> (maybe) SUGGEST -> log
        final_state_from_graph = hpo_graph.invoke(state_for_graph)

        # Always refresh analysis + search space
        client_states[client_id]['search_space']   = final_state_from_graph.get('search_space', {})
        client_states[client_id]['last_analysis']  = final_state_from_graph.get('last_analysis', {})

        # In final phase there is no next round; don't overwrite concrete_hps
        if phase == "finalize":
            print(f"[CPU Worker]: Final analysis saved for client {client_id} "
                  f"(no new HPs; final round).")
        else:
            # Normal training phase: store HPs for next round usage
            new_hps = final_state_from_graph.get('hps', {})
            if new_hps:
                client_states[client_id]['concrete_hps'] = new_hps
            print(f"[CPU Worker]: New HPs and search space for client {client_id} are ready.")

        # Complete this queue item
        results_queue.task_done()
