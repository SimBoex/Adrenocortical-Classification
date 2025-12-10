
from Components import logger
import pandas as pd
import numpy as np

Customlogger = logger.createLogger("Ingestion phase",False)

def FirstFiltering(db):
    db = db[db["tenere_finale"] == 1].copy()
    Customlogger.info("from first fitering using the tenere_finale attribute,  I obtained {} patients".format(db.shape))
    return db

'''
This function takes the dataset and extract the patients after removing the columns with too  many Nan and the ones with the attribute tenere_finale set to 1

INPUT: PATH
OUTPUT: cleaned dataset (pandas dataframe)
'''
def extract_withAnalysis(PATH):
    df = pd.read_excel(PATH)
    df1 = FirstFiltering(df)

    df_beforeRemovingColumns = df1.copy()

    obj, db3, df_notDropped, df = FilteringColumns(df1)
    return obj, db3, df_notDropped, df, df_beforeRemovingColumns


import pandas as pd

def analyze_columns(db):
    print(f"the len of the dataset is {len(db)}")
    # Count NaN values for each column
    nan_counts = db.isna().sum(axis = 0)
    nan_percentages = (nan_counts / len(db)) * 100
    
    # Identify columns with a single unique value
    single_value_columns = [col for col in db.columns if db[col].nunique() == 1]
    
    # Identify columns that contain a ":" character in any of their values
    columns_with_colon = [col for col in db.columns if db[col].astype(str).str.contains(':', na=False).any()]
    columns_with_PARAMS = [col for col in db.columns if "PARAMS" in col]
    columns_with_INFO = [col for col in db.columns if "INFO" in col]
    columns_with_TIME = [col for col in db.columns if "Time" in col]



    columns_with_bar = [col for col in db.columns if db[col].astype(str).str.contains('|', na=False, regex = False).any()]

    columns_with_xyz = [col for col in db.columns if db[col].astype(str).str.contains(r'x.*y.*z|x.*z.*y|y.*x.*z|y.*z.*x|z.*x.*y|z.*y.*x', na=False, regex=True).any()]

    columns_with_arrows = [col for col in db.columns if db[col].astype(str).str.contains('<->', na=False, regex = False).any()]

    # Create a DataFrame summarizing the results
    summary_df = pd.DataFrame({
        'NaN Count': nan_counts,
        'NaN Percentage': nan_percentages,
        'Single Value': [col in single_value_columns for col in db.columns],
        'Contains ":"': [col in columns_with_colon for col in db.columns],
        'Contain Bar':[col in columns_with_bar for col in db.columns],
        'Contain x,y, z' : [col in columns_with_xyz for col in db.columns],
        'Contain arrows': [col in columns_with_arrows for col in db.columns],
        'Contain PARAMS': [col in columns_with_PARAMS for col in db.columns],
        'Contain INFO': [col in columns_with_INFO for col in db.columns],
        'Contain TIME': [col in columns_with_TIME for col in db.columns],

    })
    
    return summary_df

def FilteringColumns(db):
    # List of columns to drop


    # Define the columns to drop (radiomics features and other unwanted columns)
    dropped_columns = [
        'BAS_INFO_ActualFrameDuration', 'BAS_INFO_NameOfRoi', 'BAS_PARAMS_DistanceOfNeighbours',
        'BAS_PARAMS_NumberOfGreyLevels', 'BAS_PARAMS_BinSize', 'BAS_PARAMS_IntensityResampling',
        'BAS_PARAMS_ZSpatialResampling', 'BAS_PARAMS_YSpatialResampling', 'BAS_PARAMS_XSpatialResampling',
        'BAS_CHECK_ClustersToSmall', 'BAS_TimePosition', 'BAS_zLocationonlyFor2DROI', 'BAS_FEATURERESULTS',
        'BAS_PARAMS_BoundsRangeOfValueAfterDiscretisationHU', 'BAS_INFO_ProcessDateOfTexture',
        'BAS_INTENSITYBASEDRIM_MinHUIBSINo', 'BAS_INTENSITYBASEDRIM_MeanHUIBSINo', 'BAS_INTENSITYBASEDRIM_StdevHUIBSINo',
        'BAS_INTENSITYBASEDRIM_MaxHUIBSINo', 'BAS_INTENSITYBASEDRIM_CountingVoxels#vxIBSINo',
        'BAS_INTENSITYBASEDRIM_ApproximateVolumemLIBSINo', 'BAS_INTENSITYBASEDRIM_SumHUIBSINo',
        'BAS_MORPHOLOGICAL_MaxValueCoordinatesIBSINo', 'BAS_MORPHOLOGICAL_CenterOfMassIBSINo',
        'BAS_MORPHOLOGICAL_WeightedCenterOfMassIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_MinHUIBSINo',
        'BAS_INTENSITYHISTOGRAMRIM_MeanHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_StdevHUIBSINo',
        'BAS_INTENSITYHISTOGRAMRIM_MaxHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_CountingVoxels#vxIBSINo',
        'BAS_INTENSITYHISTOGRAMRIM_ApproximateVolumemLIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_SumHUIBSINo',
        'Sesso', 'età', 'secrezionebis', 'mmasse1', 'HUpreCONTRASTO', 'tenere_finale',
        'BAS_MORPHOLOGICAL_HocIBSINo', 'BAS_MORPHOLOGICAL_NormalizedHocRadiusRoiIBSINo',
        'BAS_MORPHOLOGICAL_NormalizedHocRadiusSphereIBSINo',
        'BAS_MORPHOLOGICAL_NormalizedCentreOfMassShiftMaxRadiusRoiIBSINo',
        'BAS_MORPHOLOGICAL_NormalizedCentreOfMassShiftRadiusSphereIBSINo',
        'BAS_MORPHOLOGICAL_SphereDiameterIBSINo',
        'BAS_INTENSITYBASED_AreaUnderCurveCshHUIBSINo',
        'BAS_LOCAL_INTENSITY_BASED_IntensityPeakDiscretizedVolumeSought0',
        'BAS_LOCAL_INTENSITY_BASED_GlobalIntensityPeak0.5mLHUIBSINo',
        'BAS_LOCAL_INTENSITY_BASED_IntensityPeakDiscretizedVolumeSought1m',
        'BAS_LOCAL_INTENSITY_BASED_GlobalIntensityPeak1mLHUIBSI0F91',
        'BAS_LOCAL_INTENSITY_BASED_LocalIntensityPeakHUIBSIVJGA',
        'BAS_LOCAL_INTENSITY_HISTOGRAM_IntensityPeakDiscretizedVolumeSoug',
        'BAS_LOCAL_INTENSITY_HISTOGRAM_GlobalIntensityPeak0.5mLHUIBSINo',
        'BAS_LOCAL_INTENSITY_HISTOGRAM_IntensityPeakDiscretizedVolumeSo_A',
        'BAS_LOCAL_INTENSITY_HISTOGRAM_GlobalIntensityPeak1mLHUIBSINo',
        'BAS_LOCAL_INTENSITY_HISTOGRAM_LocalIntensityPeakHUIBSINo', 'Unnamed: 183'
    ]
    print(f"the dropped colums are {len(dropped_columns)}")
    db2 = db[dropped_columns]
    # Analysie the parameters and the radiomic with more than 50%
    obj = analyze_columns(db2)

    # let's drop the columns 
    df = db.copy()
    df = df.drop(columns=dropped_columns)
    # let's analyse the dataset cleaned (no columns with more than 50 and parameter extraction) with still NaN
    db3 = analyze_columns(df)

    return obj, db3, df, df.dropna()


