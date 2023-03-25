# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

# dat = pd.read_csv('solar_clean_data.csv')

# dat.head(10)
# print(dat)

def load_data(data,drop:list=[],take_only:list=[]):
    dat = pd.read_csv(data)
    print('Printing data preview ------>\n',dat.head(10))
    if drop:
        print('drop')
        dat.drop(drop,axis=1,inplace=True)
        print('Data preview after dropping features ------>\n',dat.head(10))
    elif take_only:
        print('takeonly')
        dat=dat[take_only]
        print('Data preview after takeing only features ------>\n',dat.head(10))
    else:
        print('else')
        dat=dat
    dat.dropna(inplace=True)
    print('data info ------>\n',dat.info())
    return dat

def factor(dat):
    chi_square_value,p_value=calculate_bartlett_sphericity(dat)
    print('chi square value:',chi_square_value)
    print('p value:',p_value)
    kmo_all,kmo_model=calculate_kmo(dat)
    print('calculated kmo:',kmo_model)
    # Create factor analysis object and perform factor analysis
    fa = FactorAnalyzer(rotation=None,svd_method='randomized')
    fa.fit(dat, 4)
    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()
    print('eigen values:',ev)

    plt.scatter(range(1,dat.shape[1]+1),ev)
    plt.plot(range(1,dat.shape[1]+1),ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

    fa = FactorAnalyzer(rotation="varimax")
    fa.fit(dat, 3)

    res = fa.loadings_
    print('factors:\n',res)
    df = pd.DataFrame(res, columns = ['factor1','factor2','factor3'], index=dat.columns)
    print(df)

    dfvar = pd.DataFrame(fa.get_factor_variance(), columns = ['factor1','factor2','factor3'], index=['SS Loadings','Proportion Var','Cumulative Var'])
    print(dfvar)

    #Calculate the communalities, given the factor loading matrix
    comm = fa.get_communalities()
    print('communalities:\n',comm)
    
    return res,df,dfvar,comm

#dat = load_data('out_clean.csv',['Insolation', 'Internet', 'Date'],['Total Generation (KWH)','Grid Failure','Module Cleaning','Cloudy','No Module Cleaning'])
#dat = load_data('out_clean.csv',['Insolation', 'Internet', 'Date'])
dat = load_data('out_clean.csv',take_only=['Insolation','Inverter','Total Generation (KWH)','Grid Failure','Module Cleaning','Cloudy','Rainy day'])
#dat = load_data('out_clean.csv',['Date','Internet'])
res,df,dfvar,comm = factor(dat)
