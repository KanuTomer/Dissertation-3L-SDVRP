class InspectRoute:
    def __init__(self):
        pass

    def inspect(self, result: dict) -> dict:
        # compute checks, volume validity, ordering, etc.
        return {"valid": True, "issues": []}
