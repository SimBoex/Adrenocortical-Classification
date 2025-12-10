
from Components import logger
import pandas as pd
import numpy as np

Customlogger = logger.createLogger("Ingestion phase",False)

def FirstFiltering(db):
    db = db[db["tenere_finale"] == 1].copy()
    Customlogger.info("from first fitering using the tenere_finale attribute,  I obtained {} patients".format(db.shape))
    return db

def FilteringColumns(db):
    dropped_columns =[
        
    'BAS_INFO_ActualFrameDuration','BAS_INFO_NameOfRoi','BAS_PARAMS_DistanceOfNeighbours','BAS_PARAMS_NumberOfGreyLevels','BAS_PARAMS_BinSize',                                             
    'BAS_PARAMS_IntensityResampling','BAS_PARAMS_ZSpatialResampling','BAS_PARAMS_YSpatialResampling','BAS_PARAMS_XSpatialResampling','BAS_CHECK_ClustersToSmall','BAS_TimePosition','BAS_zLocationonlyFor2DROI','BAS_FEATURERESULTS',                                     
    'BAS_PARAMS_BoundsRangeOfValueAfterDiscretisationHU', 'BAS_INFO_ProcessDateOfTexture',

    'BAS_INTENSITYBASEDRIM_MinHUIBSINo','BAS_INTENSITYBASEDRIM_MeanHUIBSINo', 'BAS_INTENSITYBASEDRIM_StdevHUIBSINo','BAS_INTENSITYBASEDRIM_MaxHUIBSINo', 'BAS_INTENSITYBASEDRIM_CountingVoxels#vxIBSINo', 'BAS_INTENSITYBASEDRIM_ApproximateVolumemLIBSINo',
    'BAS_INTENSITYBASEDRIM_SumHUIBSINo',
    
    'BAS_MORPHOLOGICAL_MaxValueCoordinatesIBSINo','BAS_MORPHOLOGICAL_CenterOfMassIBSINo','BAS_MORPHOLOGICAL_WeightedCenterOfMassIBSINo',
    
    'BAS_INTENSITYHISTOGRAMRIM_MinHUIBSINo','BAS_INTENSITYHISTOGRAMRIM_MeanHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_StdevHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_MaxHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_CountingVoxels#vxIBSINo',
    'BAS_INTENSITYHISTOGRAMRIM_ApproximateVolumemLIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_SumHUIBSINo',
    


    'Sesso','età','luogoTC_codificato','secrezionebis','mmasse1','HUpreCONTRASTO','tenere_finale',

    'BAS_MORPHOLOGICAL_HocIBSINo',
    'BAS_MORPHOLOGICAL_NormalizedHocRadiusRoiIBSINo',
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
    'BAS_LOCAL_INTENSITY_HISTOGRAM_LocalIntensityPeakHUIBSINo', 
    'Unnamed: 183'
    ]
    
    db = db.drop(dropped_columns, axis=1)
    db = db.dropna()
    Customlogger.info("from removing columns with too many Nans,I obtained {} patients".format(db.shape))
    db.to_excel("Dataset/Cleaned_dataset.xlsx", index=False)
    return db

def FilteringColumns_withLuogoTc(db):
    dropped_columns =[
        
    'BAS_INFO_ActualFrameDuration','BAS_INFO_NameOfRoi','BAS_PARAMS_DistanceOfNeighbours','BAS_PARAMS_NumberOfGreyLevels','BAS_PARAMS_BinSize',                                             
    'BAS_PARAMS_IntensityResampling','BAS_PARAMS_ZSpatialResampling','BAS_PARAMS_YSpatialResampling','BAS_PARAMS_XSpatialResampling','BAS_CHECK_ClustersToSmall','BAS_TimePosition','BAS_zLocationonlyFor2DROI','BAS_FEATURERESULTS',                                     
    'BAS_PARAMS_BoundsRangeOfValueAfterDiscretisationHU', 'BAS_INFO_ProcessDateOfTexture',

    'BAS_INTENSITYBASEDRIM_MinHUIBSINo','BAS_INTENSITYBASEDRIM_MeanHUIBSINo', 'BAS_INTENSITYBASEDRIM_StdevHUIBSINo','BAS_INTENSITYBASEDRIM_MaxHUIBSINo', 'BAS_INTENSITYBASEDRIM_CountingVoxels#vxIBSINo', 'BAS_INTENSITYBASEDRIM_ApproximateVolumemLIBSINo',
    'BAS_INTENSITYBASEDRIM_SumHUIBSINo',
    
    'BAS_MORPHOLOGICAL_MaxValueCoordinatesIBSINo','BAS_MORPHOLOGICAL_CenterOfMassIBSINo','BAS_MORPHOLOGICAL_WeightedCenterOfMassIBSINo',
    
    'BAS_INTENSITYHISTOGRAMRIM_MinHUIBSINo','BAS_INTENSITYHISTOGRAMRIM_MeanHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_StdevHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_MaxHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_CountingVoxels#vxIBSINo',
    'BAS_INTENSITYHISTOGRAMRIM_ApproximateVolumemLIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_SumHUIBSINo',
    


    'Sesso','età','secrezionebis','mmasse1','HUpreCONTRASTO','tenere_finale',

    'BAS_MORPHOLOGICAL_HocIBSINo',
    'BAS_MORPHOLOGICAL_NormalizedHocRadiusRoiIBSINo',
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
    'BAS_LOCAL_INTENSITY_HISTOGRAM_LocalIntensityPeakHUIBSINo', 
    'Unnamed: 183'
    ]
    
    db = db.drop(dropped_columns, axis=1)
    db = db.dropna()
    Customlogger.info("from removing columns with too many Nans,I obtained {} patients".format(db.shape))
    #db.to_excel("Dataset/Cleaned_dataset.xlsx", index=False)
    return db


def FilteringColumns_withLuogoTc_2(db):
    dropped_columns =[
        
    'BAS_INFO_ActualFrameDuration','BAS_INFO_NameOfRoi','BAS_PARAMS_DistanceOfNeighbours','BAS_PARAMS_NumberOfGreyLevels','BAS_PARAMS_BinSize',                                             
    'BAS_PARAMS_IntensityResampling','BAS_PARAMS_ZSpatialResampling','BAS_PARAMS_YSpatialResampling','BAS_PARAMS_XSpatialResampling','BAS_CHECK_ClustersToSmall','BAS_TimePosition','BAS_zLocationonlyFor2DROI','BAS_FEATURERESULTS',                                     
    'BAS_PARAMS_BoundsRangeOfValueAfterDiscretisationHU', 'BAS_INFO_ProcessDateOfTexture',

    'BAS_INTENSITYBASEDRIM_MinHUIBSINo','BAS_INTENSITYBASEDRIM_MeanHUIBSINo', 'BAS_INTENSITYBASEDRIM_StdevHUIBSINo','BAS_INTENSITYBASEDRIM_MaxHUIBSINo', 'BAS_INTENSITYBASEDRIM_CountingVoxels#vxIBSINo', 'BAS_INTENSITYBASEDRIM_ApproximateVolumemLIBSINo',
    'BAS_INTENSITYBASEDRIM_SumHUIBSINo',
    
    'BAS_MORPHOLOGICAL_MaxValueCoordinatesIBSINo','BAS_MORPHOLOGICAL_CenterOfMassIBSINo','BAS_MORPHOLOGICAL_WeightedCenterOfMassIBSINo',
    
    'BAS_INTENSITYHISTOGRAMRIM_MinHUIBSINo','BAS_INTENSITYHISTOGRAMRIM_MeanHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_StdevHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_MaxHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_CountingVoxels#vxIBSINo',
    'BAS_INTENSITYHISTOGRAMRIM_ApproximateVolumemLIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_SumHUIBSINo',
    


    'Sesso','età','secrezionebis','tenere_finale',

    'BAS_MORPHOLOGICAL_HocIBSINo',
    'BAS_MORPHOLOGICAL_NormalizedHocRadiusRoiIBSINo',
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
    'BAS_LOCAL_INTENSITY_HISTOGRAM_LocalIntensityPeakHUIBSINo', 
    'Unnamed: 183'
    ]
    
    db = db.drop(dropped_columns, axis=1)

    
    # Count the number of NaN values in column 'A' and column 'B'
    nan_A = db['mmasse1'].value_counts(dropna=False).get(np.nan, 0)
    nan_B = db['HUpreCONTRASTO'].value_counts(dropna=False).get(np.nan, 0)

    # Seleziona i soggetti con NaN in HUpreCONTRASTO
    nan_subjects = db[db['HUpreCONTRASTO'].isna()]

    # Conta i NaN per ciascuna riga (escludendo HUpreCONTRASTO)
    nan_counts = nan_subjects.isna().sum(axis=1) - 1  # escludiamo HUpreCONTRASTO stesso

    # Aggiungiamo questa informazione come nuova colonna
    nan_subjects = nan_subjects.copy()
    nan_subjects['Altri_NaN'] = nan_counts

    # Visualizza i risultati
    print(nan_subjects[['ID', 'HUpreCONTRASTO', 'Altri_NaN']])

    # Print the results
    print(f"NaN values in column mmasse1: {nan_A}")
    print(f"NaN values in column HUpreCONTRASTO: {nan_B}") 
    print(db.duplicated().sum())  # Check if there are duplicate rows

    db = db.dropna()
    
    Customlogger.info("from removing columns with too many Nans,I obtained {} patients".format(db.shape))

    return db


def FilteringColumns_withLuogoTcAndOnly2(db):
    db = FilteringColumns_withLuogoTc_2(db)
   
    # Write NaN counts to a text file
    #with open('nan_counts.txt', 'w') as file:
        #file.write(db.isna().sum().to_string())
    db = db[['ID','morf_codificata','maligno','luogoTC_codificato','mmasse1','HUpreCONTRASTO']]
    Customlogger.info("from selecting the chosen columns,I obtained {} patients".format(db.shape))
    db.to_excel("Dataset/OnlyTwoFeaturesDataset.xlsx", index=False)
    return db

def extractTwoFeatures(PATH):
    df = pd.read_excel(PATH)
    df1 = FirstFiltering(df)
    db = FilteringColumns_withLuogoTcAndOnly2(df1)
    return db


'''
This function takes the dataset and extract the patients after removing the columns with too  many Nan and the ones with the attribute tenere_finale set to 1

INPUT: PATH
OUTPUT: cleaned dataset (pandas dataframe)
'''
def extract(PATH):
    df = pd.read_excel(PATH)
    df1 = FirstFiltering(df)
    db = FilteringColumns_withLuogoTc(df1)
    return db
    
import re

def FilteringColumns_withLuogoTcAndFirstOrder(df):
    # Define regex patterns for different families
    patterns = {
        'Intensity Based': re.compile(r'BAS_INTENSITYBASED_'),
        'Intensity Histogram': re.compile(r'BAS_INTENSITYHISTOGRAM_'),
    }

    selected_columns = [
    col for col in df.columns
    if any(pattern.match(col) for pattern in patterns.values())
    ]

    append = ['ID','morf_codificata','maligno','luogoTC_codificato'] 
    cols = append + selected_columns
    print(cols)
    df2 = df[cols]


    # Count the number of NaN values in each column
    #nan_count = df2.isna().sum()
    #print(nan_count)


    df2 = df2.dropna()
    df2.to_excel("Dataset/OnlyFirstOrder.xlsx", index=False)

    print("the dataset has shape ", df2.shape)
    return df2


def extractFirstOrder(PATH):
    df = pd.read_excel(PATH)
    df1 = FirstFiltering(df)
    df1 = FilteringColumns_withLuogoTc(df1)
    db = FilteringColumns_withLuogoTcAndFirstOrder(df1)
    return db




def FilteringColumns_Secretion(db):
    dropped_columns =[
        'BAS_INFO_ActualFrameDuration','BAS_INFO_NameOfRoi','BAS_PARAMS_DistanceOfNeighbours','BAS_PARAMS_NumberOfGreyLevels','BAS_PARAMS_BinSize',                                             
    'BAS_PARAMS_IntensityResampling','BAS_PARAMS_ZSpatialResampling','BAS_PARAMS_YSpatialResampling','BAS_PARAMS_XSpatialResampling','BAS_CHECK_ClustersToSmall','BAS_TimePosition','BAS_zLocationonlyFor2DROI','BAS_FEATURERESULTS',                                     
    'BAS_PARAMS_BoundsRangeOfValueAfterDiscretisationHU', 'BAS_INFO_ProcessDateOfTexture',

    'Sesso','età','luogoTC_codificato','mmasse1','HUpreCONTRASTO','tenere_finale',

    'BAS_MORPHOLOGICAL_HocIBSINo',
    'BAS_MORPHOLOGICAL_NormalizedHocRadiusRoiIBSINo',
    'BAS_MORPHOLOGICAL_NormalizedHocRadiusSphereIBSINo',
    'BAS_MORPHOLOGICAL_NormalizedCentreOfMassShiftMaxRadiusRoiIBSINo',
    'BAS_MORPHOLOGICAL_NormalizedCentreOfMassShiftRadiusSphereIBSINo',
    'BAS_MORPHOLOGICAL_SphereDiameterIBSINo',  'BAS_INTENSITYBASEDRIM_ApproximateVolumemLIBSINo',
    'BAS_INTENSITYBASED_AreaUnderCurveCshHUIBSINo',
    'BAS_LOCAL_INTENSITY_BASED_IntensityPeakDiscretizedVolumeSought0',
    'BAS_LOCAL_INTENSITY_BASED_GlobalIntensityPeak0.5mLHUIBSINo',
    'BAS_LOCAL_INTENSITY_BASED_IntensityPeakDiscretizedVolumeSought1m',
    'BAS_LOCAL_INTENSITY_BASED_GlobalIntensityPeak1mLHUIBSI0F91',  
    'BAS_LOCAL_INTENSITY_BASED_LocalIntensityPeakHUIBSIVJGA',

    'BAS_MORPHOLOGICAL_MaxValueCoordinatesIBSINo', 'BAS_MORPHOLOGICAL_CenterOfMassIBSINo','BAS_MORPHOLOGICAL_WeightedCenterOfMassIBSINo',
    'BAS_INTENSITYBASEDRIM_MinHUIBSINo','BAS_INTENSITYBASEDRIM_MeanHUIBSINo','BAS_INTENSITYBASEDRIM_StdevHUIBSINo',
    'BAS_INTENSITYBASEDRIM_MaxHUIBSINo', 'BAS_INTENSITYBASEDRIM_CountingVoxels#vxIBSINo',
    'BAS_INTENSITYBASEDRIM_SumHUIBSINo',
    'BAS_INTENSITYHISTOGRAMRIM_MinHUIBSINo'	, 'BAS_INTENSITYHISTOGRAMRIM_MeanHUIBSINo',
    'BAS_INTENSITYHISTOGRAMRIM_StdevHUIBSINo', 'BAS_INTENSITYHISTOGRAMRIM_MaxHUIBSINo',
    'BAS_INTENSITYHISTOGRAMRIM_CountingVoxels#vxIBSINo', 	'BAS_INTENSITYHISTOGRAMRIM_ApproximateVolumemLIBSINo',	
    'BAS_INTENSITYHISTOGRAMRIM_SumHUIBSINo',
                            
    'BAS_LOCAL_INTENSITY_HISTOGRAM_IntensityPeakDiscretizedVolumeSoug',
    'BAS_LOCAL_INTENSITY_HISTOGRAM_GlobalIntensityPeak0.5mLHUIBSINo',
    'BAS_LOCAL_INTENSITY_HISTOGRAM_IntensityPeakDiscretizedVolumeSo_A',
    'BAS_LOCAL_INTENSITY_HISTOGRAM_GlobalIntensityPeak1mLHUIBSINo',
    'BAS_LOCAL_INTENSITY_HISTOGRAM_LocalIntensityPeakHUIBSINo', 
    'Unnamed: 183'
    ]
    db = db.drop(dropped_columns, axis=1)
    db = db.dropna()
    Customlogger.info("from removing columns with too many Nans,I obtained {} patients".format(db.shape))
    
    encoded_df = pd.get_dummies(db['secrezionebis'], prefix='secrezione', dummy_na=False)
    encoded_df = encoded_df.rename(columns={'secrezione_0.0': 'secrezione_0', 'secrezione_1.0': 'secrezione_1','secrezione_2.0': 'secrezione_2','secrezione_3.0': 'secrezione_3','secrezione_4.0': 'secrezione_4','secrezione_5.0': 'secrezione_5','secrezione_6.0': 'secrezione_6','secrezione_7.0': 'secrezione_7', 'secrezione_8.0': 'secrezione_8','secrezione_9.0': 'secrezione_9','secrezione_10.0': 'secrezione_10'})
    encoded_df = encoded_df.astype(int)
    
    new_df_secretion = pd.concat([db, encoded_df], axis=1)
    del new_df_secretion["secrezionebis"]
    
    # removing class 4 due to a bias (patients with 4 were mainly not tested for secretion)
    new_df_secretion =  new_df_secretion[new_df_secretion["morf_codificata"]!=4]
    
    new_df_secretion.to_excel("Dataset/IntegratingSecretion_Cleaned_dataset.xlsx", index=False)
    return  new_df_secretion

def extract_integrating_Secretion(PATH):
    df = pd.read_excel(PATH)
    df1 = FirstFiltering(df)
    db = FilteringColumns_Secretion(df1)
    return db