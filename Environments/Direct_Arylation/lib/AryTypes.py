from dataclasses import dataclass
# from Environments.Direct_Arylation.lib.utils import BASE_DICT_REVERSED, SOLVENT_DICT_REVERSED

@dataclass
class ExperimentProperty:
    Temperature: float
    Concentration: float
    Base: float
    Solvent: float
    reation_yield: tuple

    def __str__(self) -> str:

        result_str = '\tProcess Condition\n'
        result_str += f'Temperature: {self.Temperature:.4f}\n'
        result_str += f'Concentration: {self.Concentration:.4f}\n'
        result_str += f'Base: {self.Base}\n'
        result_str += f'Solvent: {self.Solvent}\n'
        result_str += '\tPrediction Result\n'
        result_str += f'Reation Yield: {self.reation_yield[0]:.4f}, {self.reation_yield[1]:.4f}\n'
        result_str += '-------------------------'
        return result_str