from ..isolators.iterative_isolate import IterativeIsolator

def run_isolator(config: dict, result: dict):
    isolator = IterativeIsolator(config)
    return isolator.isolate(result)
