from ops.services import Result
import time

def handle_classification_results(results: Result):
    if results.batch % 200 == 0:
        print(f'Batch: {results.batch}, Loss: {results.loss:.4f}')
