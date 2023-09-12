from dataclasses import dataclass
class Recipts:
    combinations = {}
    def get_pair(self, ds: str, al_metric: str, model_name:str):
        return self.combinations.get((ds, al_metric, model_name), None)
