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
        """
        result_str = '\tMolecular Property\n'
        try:
            result_str += f'SpAbs_A: {self.SpAbs_A:.4f}\n'
        except:
            result_str += f'SpAbs_A:: None Value\n'

        try:
            result_str += f'AATS0dv: {self.AATS0dv:.4f}\n'
        except:
            result_str += f'AATS0dv: None Value\n'

        try:
            result_str += f'ABC: {self.ABC:.4f}\n'
        except:
            result_str += f'ABC: None Value\n'
        """

        result_str = '\tProcess Condition\n'
        result_str += f'Temperature: {self.Temperature:.4f}\n'
        result_str += f'Concentration: {self.Concentration:.4f}\n'
        # result_str += f'Base: {BASE_DICT_REVERSED[int(self.Base)]}\n'
        # result_str += f'Solvent: {SOLVENT_DICT_REVERSED[int(self.Solvent)]}\n'
        result_str += f'Base: {self.Base}\n'
        result_str += f'Solvent: {self.Solvent}\n'
        result_str += '\tPrediction Result\n'
        result_str += f'Reation Yield: {self.reation_yield[0]:.4f}, {self.reation_yield[1]:.4f}\n'
        result_str += '-------------------------'
        return result_str