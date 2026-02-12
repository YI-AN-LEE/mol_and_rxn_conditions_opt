from dataclasses import dataclass

# @dataclass
# class MoleculeProperty:
#     Reagent1: float
#     Reagent2: float
#     Reagent3: float
#     Reagent4: float
#     lab_code: int
#     AATS3i: float
#     ATSC5Z: float
#     AATSC5Z : float
#     crystal_size: tuple

#     def __str__(self) -> str:
#         result_str = '\tMolecular Property\n'
#         try:
#             result_str += f'AATS3i: {self.AATS3i:.4f}\n'
#         except:
#             result_str += f'None Value\n'

#         try:
#             result_str += f'ATSC5Z: {self.ATSC5Z:.4f}\n'
#         except:
#             result_str += f'None Value\n'

#         try:
#             result_str += f'MATS5d: {self.AATSC5Z:.4f}\n'
#         except:
#             result_str += f'None Value\n'

#         result_str += '\tProcess Condition\n'
#         result_str += f'Reagent1 (ul): {self.Reagent1:.4f}\n'
#         result_str += f'Reagent2 (ul): {self.Reagent2:.4f}\n'
#         result_str += f'Reagent3 (ul): {self.Reagent3:.4f}\n'
#         result_str += f'Reagent4 (ul): {self.Reagent4:.4f}\n'
#         result_str += f'lab_code: {self.lab_code}\n'
#         result_str += '\tPrediction Result\n'
#         result_str += f'crystal_size: {self.crystal_size[0]:.4f}, {self.crystal_size[1]:.4f}\n'
#         result_str += '-------------------------'
#         return result_str
    
@dataclass
class ExperimentProperty:
    Reagent1: float
    Reagent2: float
    Reagent3: float
    Reagent4: float
    lab_code: int
    #ATSC5v: float
    #AATSC5Z: float
    #MATS8se : float
    crystal_size: tuple

    def __str__(self) -> str:
        """
        result_str = '\tMolecular Property\n'
        try:
            result_str += f'ATSC5v: {self.ATSC5v:.4f}\n'
        except:
            result_str += f'None Value\n'

        try:
            result_str += f'AATSC5Z: {self.AATSC5Z:.4f}\n'
        except:
            result_str += f'None Value\n'

        try:
            result_str += f'MATS8se: {self.MATS8se:.4f}\n'
        except:
            result_str += f'None Value\n'
        """

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


# 2024  2   20  'ATSC5v', 'AATSC5Z', 'MATS8se'

# @dataclass
# class SampleProperty:
#     Reagent1: float
#     Reagent2: float
#     Reagent3: float
#     Reagent4: float
#     lab_code: int
#     AATS3i: float
#     ATSC5Z: float
#     MATS5d : float
#     crystal_size: float

#     def __str__(self) -> str:
#         return f'''\tMolecular Procerty
# AATS3i: {self.AATS3i:.4f}
# ATSC5Z: {self.ATSC5Z:.4f}
# MATS5d: {self.MATS5d:.4f}
# \tPrcoess Condition
# Reagent1 (ul): {self.Reagent1:.4f}
# Reagent2 (ul): {self.Reagent2:.4f}
# Reagent3 (ul): {self.Reagent3:.4f}
# Reagent4 (ul): {self.Reagent4:.4f}
# lab_code: {self.lab_code}
# \tPrediction Result
# crystal_size: {self.crystal_size:.4f}
# -------------------------'''

# crystal_score: tuple

# crystal_score: {self.crystal_score[0]:.4f}, {self.crystal_score[1]:.4f}






# # ['Reagent1 (ul)','Reagent2 (ul)','Reagent3 (ul)','Reagent4 (ul)','lab_code','AATS3i','ATSC5v','AATSC5Z']

# # 建立 ExperimentProperty 實例
# experiment_property = ExperimentProperty(
#     ATSC5v=0.1234,
#     AATSC5Z=0.5678,
#     MATS8se=0.9012,
#     crystal_size=(100.0, 200.0),
#     # crystal_score=(0.8, 0.9),
#     Reagent1=10.0,
#     Reagent2=20.0,
#     Reagent3=30.0,
#     Reagent4=40.0,
#     lab_code=123456
# )

# # 列印 ExperimentProperty 實例
# print(experiment_property.crystal_size[0])