from dataclasses import dataclass

@dataclass
class ExperimentProperty:
    Reagent1: float
    Reagent2: float
    Reagent3: float
    Reagent4: float
    lab_code: int
    crystal_size: tuple

    def __str__(self) -> str:
        result_str = '\tProcess Condition\n'
        result_str += f'Reagent1 (ul): {self.Reagent1:.4f}\n'
        result_str += f'Reagent2 (ul): {self.Reagent2:.4f}\n'
        result_str += f'Reagent3 (ul): {self.Reagent3:.4f}\n'
        result_str += f'Reagent4 (ul): {self.Reagent4:.4f}\n'
        result_str += f'lab_code: {self.lab_code}\n'
        result_str += '\tPrediction Result\n'
        result_str += f'crystal_size: {self.crystal_size[0]:.4f}, {self.crystal_size[1]:.4f}\n'
        result_str += '-------------------------'
        return result_str

