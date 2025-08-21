import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from hyperopt import hp, Trials, fmin, tpe
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance

def descriptor_matrix(molecule_index, lookup_table, lookup='SMILES', name=''):
    """Generate a descriptor matrix."""
    
    # New column names
    
    columns = list(lookup_table.columns.values)
    new_columns = []
    for column in columns:
        if name != '':
            new_columns.append(name + '_' + str(column))
        else:
            new_columns.append(column)
    
    # Build descriptor matrix
        
    build = []
    for entry in list(molecule_index):
        match = lookup_table[lookup_table[lookup] == entry]
        if len(match) > 0:
            build.append(list(match.iloc[0]))
        else:
            build.append(np.full(len(columns),np.NaN))
            
    build = pd.DataFrame(data=build, columns=new_columns)
    
    return build

def build_experiment_index(index, index_list, lookup_table_list, lookup_list):
    """Build a descriptor matrix."""

    matrix = descriptor_matrix(index_list[0], 
                               lookup_table_list[0], 
                               lookup=lookup_list[0])
    
    matrix.insert(0, 'entry', list(index))
    
    for i in range(1,len(index_list)):
        new = descriptor_matrix(index_list[i], 
                                lookup_table_list[i], 
                                lookup=lookup_list[i])
        new['entry'] = list(index)
        matrix = matrix.merge(new, on='entry')
    
    return matrix

def load_arylation_data(base='ohe', ligand='ohe', solvent='ohe'):
    """
    Load direct arylation data with different features.
    """
    
    # SMILES index
    index = pd.read_csv('data/experiment_index.csv')
    
    # Base features
    
    if base == 'dft':    
        base_features = pd.read_csv('data/base_dft.csv')
    elif base == 'mordred':
        base_features = pd.read_csv('data/base_mordred.csv')
    elif base == 'ohe':
        base_features = pd.read_csv('data/base_ohe.csv')
        
    # Ligand features    
          
    if ligand == 'random-dft':
        ligand_features = pd.read_csv('data/ligand-random_dft.csv')   
    elif ligand == 'boltzmann-dft':
        ligand_features = pd.read_csv('data/ligand-boltzmann_dft.csv')
    elif ligand == 'mordred':       
        ligand_features = pd.read_csv('data/ligand_mordred.csv')
    elif ligand == 'ohe':
        ligand_features = pd.read_csv('data/ligand_ohe.csv')
        
    # Solvent features
    
    if solvent == 'dft':
        solvent_features = pd.read_csv('data/solvent_dft.csv')
    elif solvent == 'mordred':
        solvent_features = pd.read_csv('data/solvent_mordred.csv')
    elif solvent == 'ohe':
        solvent_features = pd.read_csv('data/solvent_ohe.csv')
        
    # Build the descriptor set
    
    index_list = [index['Base_SMILES'],
                  index['Ligand_SMILES'],
                  index['Solvent_SMILES']]
    
    lookup_table_list = [base_features, 
                         ligand_features,
                         solvent_features]
    
    lookup_list = ['base_SMILES', 
                   'ligand_SMILES',
                   'solvent_SMILES']

    experiment_index = build_experiment_index(index['entry'], 
                                              index_list, 
                                              lookup_table_list,
                                              lookup_list)

    experiment_index['concentration'] = index['Concentration']
    experiment_index['temperature'] = index['Temp_C']
    experiment_index['yield'] = index['yield']
    
    return experiment_index

def hyperopt_objective(params, model_type):
    if model_type == 'xgb':
        model = XGBRegressor(max_depth=int(params['max_depth']),
                        n_estimators=int(params['n_estimators']),
                        gamma=params['gamma'],
                        learning_rate=params['learning_rate'],
                        )
        
    if model_type == 'rfr':
        model = RandomForestRegressor(n_estimators=int(params['n_estimators']),
                                   max_depth=int(params['max_depth']),
                                   min_samples_split=int(params['min_samples_split']),
                                   min_samples_leaf=int(params['min_samples_leaf']),
                                   )
    
    return -1 * cross_val_score(model, X, y, cv=10).mean()

def param_hyperopt(params, model_type, max_eval=100):
    trials = Trials()

    early_stop = no_progress_loss(100)

    # 使用 functools.partial() 函數來固定 hyperopt_objective() 函數的 model_type 參數
    objective_func = partial(hyperopt_objective, model_type=model_type)

    best_params = fmin(fn=objective_func,
                space=params,
                algo=tpe.suggest,
                max_evals=max_eval,
                verbose=True,
                trials=trials,
                early_stop_fn=early_stop,)

    return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for XGBoost.")
    parser.add_argument('--seed', type=int, help='random seed for reproducibility', default=42)
    parser.add_argument('--drop_rule', type=int, help='0 for axis=0, which is default. 1 for axis=1, whcih is delete the wrong column. ', default=0)
    parser.add_argument('--model_type', type=str, help='regressoion model type', required=True)
    parser.add_argument('--feature', type=str, help='molecular feature type', required=True)

    args = parser.parse_args()
    
    print('seed:', args.seed)
    np.random.seed(args.seed)
    if args.feature == 'mordred':
        reaction_data = load_arylation_data(base='mordred', ligand='mordred', solvent='mordred')
    elif args.feature == 'dft':
        reaction_data = load_arylation_data(base='dft', ligand='boltzmann-dft', solvent='dft')
    reaction_data = reaction_data.sample(frac=1, random_state=args.seed)

    print("Unique data types in the dataset:")
    print(reaction_data.dtypes.unique())

    print("All columns of type 'object' in the dataset:\n")
    for column in reaction_data.select_dtypes(include=['object']).columns:
        print(f"{column}:")

    reaction_data = reaction_data.select_dtypes(include=['int64', 'float64'])
    if args.drop_rule == 0:
        reaction_data = reaction_data.dropna()
    elif args.drop_rule == 1:
        reaction_data = reaction_data.dropna(axis=1) # usually for modred
    else:
        raise ValueError('drop_rule should be 0 or 1')
    
    reaction_data = reaction_data.drop('entry', axis=1)
    X = reaction_data.iloc[:, :-1]
    y = reaction_data.iloc[:, -1]
    X.columns = X.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_')


    nan_columns = reaction_data.columns[reaction_data.isnull().any()].tolist()
    print("Columns with NaN values:", nan_columns)

    inf_columns = reaction_data.columns[(reaction_data == np.inf).any()].tolist()
    print("Columns with infinity values:", inf_columns)

    large_value_columns = reaction_data.columns[(reaction_data > np.finfo(np.float32).max).any()].tolist()
    print("Columns with values too large for float32:", large_value_columns)

    '''
    Random Forest Regressor
    '''
    if args.model_type == 'rfr':
        params = {
            'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
            'max_depth': hp.quniform('max_depth', 5, 30, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        }
        best_params = param_hyperopt(params, args.model_type)
        print('best parameters:', best_params)
        
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

        model = RandomForestRegressor(**best_params)
        model.fit(X, y)


        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

    elif args.model_type == 'xgb':
        params = {'max_depth': hp.quniform('max_depth', 3, 20, 1),
            'n_estimators': hp.quniform('n_estimators', 50, 150, 1),
            'gamma': hp.uniform('gamma', 0, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            }
        
        best_params = param_hyperopt(params, args.model_type)

        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        print('best_params',best_params)

        model = XGBRegressor(random_state=args.seed, **best_params)
        model.fit(X, y)

      
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
    else:
        raise ValueError('model_type should be rfr or xgb')


    plt.figure(figsize=(10, 8), dpi=400)
    plt.scatter(y, y_pred, alpha=0.8, color='#0F5257', s=20)  
    plt.ylabel('Prediction Value (Percentage)')


    limits = [min(min(y), min(y_pred)), max(max(y), max(y_pred))]
    hyp_scores = cross_val_score(model, X, y, cv=10)
    if args.model_type == 'rfr':    
        name = "Random Forest Regresssor"
    elif args.model_type == 'xgb':
        name = "XGBoost Regressor"

    if args.feature == 'mordred':
        feature_name = 'Mordred'
    elif args.feature == 'dft':
        feature_name = 'DFT'

    mean = np.mean(hyp_scores)
    std = np.std(hyp_scores)
    scores = hyp_scores
    print(f'Model: {name}\nCross-validation scores: {scores}')
    plt.plot(limits, limits, color='#0B3142')  
    plt.title(f'{name} for Yield Prediction ({feature_name} feature)')
    plt.text(0.02, 0.91, f'Regression $R^2$: {r2:.4f}\nMean cross-validation score: {mean:.4f}\nStandard deviation: {std:.4f}', transform=plt.gca().transAxes)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.savefig(f'Regression_Model_Analysis/Parity_plot_{args.model_type}_{args.feature}_{args.seed}.png')

  
    fig, ax = plt.subplots(figsize=(20, 16))

    if args.model_type == 'xgb':

        fig, ax = plt.subplots(figsize=(20, 16))

        plot_importance(model, ax=ax, importance_type='weight')

        plt.show()

    elif args.model_type == 'rfr':

        importances = model.feature_importances_
        features = X.columns
        importances_features = sorted(zip(importances, features), reverse=True)[:50]

        sorted_importances, sorted_features = zip(*importances_features)

        plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
        plt.yticks(range(len(sorted_importances)), sorted_features)
        plt.xlabel('Importance')
        plt.title('Feature Importances')

    plt.savefig(f'Regression_Model_Analysis/Feature_importance_{args.model_type}_{args.feature}_{args.seed}.png')
