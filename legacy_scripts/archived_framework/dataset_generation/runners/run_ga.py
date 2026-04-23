from ..algorithms.ga.ga_runner import GARunner

def run_ga_from_config(config: dict, data: dict):
    runner = GARunner(config)
    return runner.run(data)
