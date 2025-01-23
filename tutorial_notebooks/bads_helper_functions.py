####################################################################################
# The module bads_helper_functions.py contains a set of helper functions that are
# used in the tutorial notebooks. The functions are designed to simplify routine 
# tasks like loading and preparing a given data set to streamline the tutorial notebooks
####################################################################################

def get_HMEQ_credit_data(data_url='https://raw.githubusercontent.com/Humboldt-WI/bads/master/data/hmeq.csv', 
                         outlier_factor=0,
                         scale_features=False):
    '''
        Fetches and prepares the Home Equity (HMEQ) credit data from a predefined source.
        
        The function retrieves the HMEQ credit data, which includes various 
        attributes related to credit risk assessment. Specifically, the available features include: 
        
        - BAD: the target variable, 1=default; 0=non-default 
        - LOAN: amount of the loan request
        - MORTDUE: amount due on an existing mortgage
        - VALUE: value of current property
        - REASON: DebtCon=debt consolidation; HomeImp=home improvement
        - JOB: occupational categories
        - YOJ: years at present job
        - DEROG: number of major derogatory reports
        - DELINQ: number of delinquent credit lines
        - CLAGE: age of oldest credit line in months
        - NINQ: number of recent credit inquiries
        - CLNO: number of credit lines
        - DEBTINC: debt-to-income ratio
        
        
        The function prepares the data by performing the following steps:
        - Type conversion of the target variable and categorical features
        - Discretization of numerical features
        - Imputation of missing values among categorical and numerical features
        - Truncation of outliers among numerical features
        - Encoding of categorical features

        Args:
            data_url (str): The URL of the data source. The default URL points 
            to the raw data file on GitHub.scale_features
                        
            outlier_factor (float): The factor used to determine the range of 
            acceptable values for numerical features by Tuckey's rule, which
            defines an upper/lower outlier to be a value outlier_factor*IQR 
            below/above the first/third quartile. The default value is -1 meaning
            that no truncation is performed.

            scale_features (bool): A binary variable indicating whether all 
            numerical features should be scaled using a z-transformation

        Returns:
            X: A pandas DataFrame containing the feature matrix of the prepared data.
            y: A pandas Series containing the binary target variable, where a value of
              'True' indicates a bad credit risk.
    '''
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    ####################################################################
    # Load data 
    ####################################################################
    df = pd.read_csv(data_url)
    ####################################################################
    # Data preparation
    ####################################################################
    # Type conversion
    #-------------------------------------------------------------------
    df['BAD'] = df['BAD'].astype('bool')  # The target variable has only two states so that we can store it as a boolean
    df['REASON'] = df['REASON'].astype('category')  # Code categories properly 
    df['JOB'] = df['JOB'].astype('category')
    #-------------------------------------------------------------------
    # Discretize numerical features DELINQcat and DEROG
    #-------------------------------------------------------------------
    df['DELINQcat'] = pd.cut(df['DELINQ'], bins=[-1, 0, 1, float('inf')], labels=['0', '1', '1+']).astype('category')
    df['DEROGzero'] = pd.cut(df['DEROG'], bins=[-1, 0, float('inf')], labels=[1, 0]).astype(bool)
    df = df.drop(columns=['DELINQ', 'DEROG'])  # Drop the original columns
    #-------------------------------------------------------------------
    # Missing values among categorical features
    #-------------------------------------------------------------------
    ix_cat = df.select_dtypes(include=['category']).columns  # Get an index of the categorical columns
    for c in ix_cat:  # Process each category
        df.loc[df[c].isna(), c ] = df[c].mode()[0]  # the index [0] is necessary as the result of calling mode() is a Pandas Series

    # Verify that there are no missing values left
    if np.any(df[ix_cat].isna()):
        raise Exception(f"We still observe {df[ix_cat].isna().sum()} missing values among the categorical features")
    #-------------------------------------------------------------------
    # Missing values among DEBTINC
    #-------------------------------------------------------------------
    # The feature DEBTINC is important but suffers many missing values. Blindly replacing these missing values
    # would introduce bias and harm any model trained on the data. To avoid this, we add a dummy variable
    # #  to indicate whether the feature value was missing or not.
    df['D2I_miss'] = df['DEBTINC'].isna().astype('category')
    #-------------------------------------------------------------------
    # Missing values among numerical features
    #-------------------------------------------------------------------
    imputer = SimpleImputer(strategy='median')  # Create an imputer object with the strategy 'median'
    ix_num = df.select_dtypes(include=np.number).columns  # Select only numerical columns
    df[ix_num] = imputer.fit_transform(df[ix_num])  # Apply the imputer to the numerical columns
    # Verify that there are no missing values left
    if np.any(df[ix_num].isna()):
        raise Exception(f"We still observe {df[ix_num].isna().sum()} missing values among the numerical features")
    #-------------------------------------------------------------------
    # Truncate outliers among numerical features
    #-------------------------------------------------------------------
    if outlier_factor > 0:
        for col in ix_num:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_factor * IQR
            upper_bound = Q3 + outlier_factor * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    #-------------------------------------------------------------------
    # Scale numerical features using the z-transformation (if requested)
    #-------------------------------------------------------------------
    if scale_features == True:
        scaler = StandardScaler()
        df[ix_num] = scaler.fit_transform(df[ix_num])
    #-------------------------------------------------------------------
    # Dummy encode categorical features
    #-------------------------------------------------------------------
    df = pd.get_dummies(df, drop_first=True)  
    #-------------------------------------------------------------------
    # Separate the target variable and the feature matrix
    #-------------------------------------------------------------------
    y = df.pop('BAD')
    X = df
    return X, y

def plot_logit_decision_surface(model, data, x1, x2, y, save_fig=False):
    '''
        Visualization of logistic regression in 2D
        
        Creates a plot depicting the distribution of the input
        data along two dimensions and the probability predictions
        of a logistic regression model. 

        Parameters
        ----------
        model :   An instance of the sklearn class LogisticRegression,  which        
                  has been trained on the input data.

        data  :   Pandas data frame providing the feature values.

        x1, x2:   The function plots the results of logistic regression in
                  two dimensions. The parameters x1 and x2 give the names
                  of the features used for plotting. These features will be
                  extracted from the data frame.

        y     :   Pandas series containing the binary target variable. 

        save_fig: Binary variable allowing you to save the figure as a PNG image. 
                  Default: False

        Returns
        ----------
        The function does not return a result. It's purpose is to visualize 
        logistic regression model. The corresponding plot is the only output.
    '''
    import numpy as np 
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    if len(model.coef_.ravel())!=2:
        raise Exception('Please estimate a logit model using only two features!')
    
    # Define some variables to govern the plot
    bounds = data.describe().loc[["min", "max"]][[x1, x2]].to_numpy()  # value ranges of the two features
    eps = 5  # tolerance parameter 

    # Create hypothetical data points spanning the entire range of feature values.
    # We need these to get from our logistic regression model a probability prediction
    # for every possible data point
    xx, yy = np.mgrid[(bounds[0,0]-eps):(bounds[1,0]+eps), (bounds[0,1]-eps):(bounds[1,1]+eps)]
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Perhaps the logistic regression model was fitted using the full data frame. 
    # To also work in that case, we extract the estimated regression coefficients 
    # corresponding to the two features we consider for plotting
    feature_to_index = {name: idx for idx, name in enumerate(model.feature_names_in_)}  # create a dic as intermediate step
    indices = [feature_to_index[f] for f in [x1, x2]]  # Find the indices of our two features of interest using the dic
    w = model.coef_.ravel()[indices]  # estimated regression coefficients
    b = model.intercept_  # estimated intercept of the logistic regression model

    # Compute probability predictions over the entire space of possible feature values
    # In the interest of robustness, we manually compute the logistic regression predictions
    # using the regression coefficients extracted above
    probs = 1/(1+np.exp(-(np.dot(grid, w.reshape(2,-1))+b))).reshape(xx.shape)

    # We are finally ready to create our visualization
    f, ax = plt.subplots(figsize=(8, 6))  # new figure
    # Contour plot of the probability predictions across the entire feature range
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)  
    ax_c = f.colorbar(contour)
    ax_c.set_label("$\hat{p}(y=1|X)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    # Scatter plot of the actual data
    ax.scatter(data[x1], data[x2], c=y, s=50, cmap="RdBu", vmin=0, vmax=1,
               edgecolor="white", linewidth=1);
    plt.xlabel(x1)
    plt.ylabel(x2)
    if save_fig==True:
        plt.savefig('logit_contour.png')
    plt.show()