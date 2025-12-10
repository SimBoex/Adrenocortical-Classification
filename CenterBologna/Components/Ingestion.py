
from Components import logger
import pandas as pd

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
    db.to_excel("Dataset/Cleaned_dataset.xlsx", index=False)

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