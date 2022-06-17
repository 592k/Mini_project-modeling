# Reducing the number of the features

import pandas as pd
import numpy as np

def featurereduce(df_train):

    # id_30 
    df_train.loc[df_train['id_30'].isin(['Windows 10',
                                        'Windows 7',
                                        'Windows 8.1',
                                        'Windows Vista',
                                        'Windows 8',
                                        'Windows XP'
                                        ]),'id_30'] = 'Windows'

    df_train.loc[df_train['id_30'].isin(['Android 7.0',
                                        'Android 5.1.1',
                                        'Android 7.1.1',
                                        'Android 6.0.1',
                                        'Android 8.0.0',
                                        'Android 4.4.2',
                                        'Android 8.1.0',
                                        'Android 6.0',
                                        'Android 7.1.2',
                                        'Android 5.0.2',
                                        'Android 5.0',
                                        ]),'id_30'] = 'Android'

    df_train.loc[df_train['id_30'].isin(['iOS 11.2.1', 
                                        'iOS 11.2.5', 
                                        'iOS 11.3.0', 
                                        'iOS 11.1.2',
                                        'iOS 11.2.2',
                                        'iOS 11.2.6',
                                        'iOS 10.3.3',
                                        'iOS 11.2.0',
                                        'iOS 11.1.1',
                                        'iOS 11.1.0',
                                        'iOS 11.0.3',
                                        'iOS 10.3.2', 
                                        'iOS 11.0.1',
                                        'iOS 10.2.1', 
                                        'iOS 10.3.1',
                                        'iOS 11.0.2',
                                        'iOS 11.0.0',
                                        'iOS 10.2.0',
                                        'iOS 9.3.5', 
                                        'iOS 10.0.2',
                                        'iOS 11.3.1',
                                        'iOS 11.4.0',
                                        'iOS 11.4.1',
                                        'iOS 10.1.1',
                                        ]),'id_30'] = 'iOS'

    df_train.loc[df_train['id_30'].isin(['Mac OS X 10_10_5',
                                        'Mac OS X 10_13_3', 
                                        'Mac OS X 10_12_6',      
                                        'Mac OS X 10_9_5',      
                                        'Mac OS X 10_13_4',      
                                        'Mac OS X 10_11_6',     
                                        'Mac OS X 10_13_2',      
                                        'Mac OS X 10_13_1',      
                                        'Mac OS X 10.12',         
                                        'Mac OS X 10_12_1',       
                                        'Mac OS X 10.10',         
                                        'Mac OS X 10_11_5',       
                                        'Mac OS X 10.11',         
                                        'Mac OS X 10_12_3',       
                                        'Mac OS X 10_12_5',       
                                        'Mac OS X 10_11_4',       
                                        'Mac OS X 10.13',
                                        'Mac OS X 10_12_4',
                                        'Mac OS X 10_7_5',
                                        'Mac OS X 10_8_5',
                                        'Mac OS X 10.9',
                                        'Mac OS X 10_11_3',
                                        'Mac OS X 10_6_8',
                                        'Mac OS X 10_12',
                                        'Mac OS X 10.6',
                                        'Mac OS X 10_12_2',
                                        'Mac OS X 10_13_5',
                                        ]),'id_30'] = 'Mac'

    df_train.loc[df_train['id_30'].isin(['func',]),'id_30'] = 'other'


    # id_31
    df_train.loc[df_train['id_31'].isin(['chrome 43.0 for android',
                                        'chrome 46.0 for android',
                                        'chrome 49.0', 
                                        'chrome 49.0 for android',
                                        'chrome 50.0',
                                        'chrome 50.0 for android',
                                        'chrome 51.0',
                                        'chrome 51.0 for android',
                                        'chrome 52.0',
                                        'chrome 52.0 for android',
                                        'chrome 53.0',
                                        'chrome 53.0 for android',
                                        'chrome 54.0',
                                        'chrome 54.0 for android',
                                        'chrome 55.0',
                                        'chrome 55.0 for android',
                                        'chrome 56.0',
                                        'chrome 56.0 for android',
                                        'chrome 57.0',
                                        'chrome 57.0 for android',
                                        'chrome 58.0',
                                        'chrome 58.0 for android',
                                        'chrome 59.0',
                                        'chrome 59.0 for android',
                                        'chrome 60.0',
                                        'chrome 60.0 for android',
                                        'chrome 61.0',
                                        'chrome 61.0 for android',
                                        'chrome 62.0',
                                        'chrome 62.0 for android',
                                        'chrome 62.0 for ios',
                                        'chrome 63.0',
                                        'chrome 63.0 for android',
                                        'chrome 63.0 for ios',
                                        'chrome 64.0',
                                        'chrome 64.0 for android',
                                        'chrome 64.0 for ios',
                                        'chrome 65.0',
                                        'chrome 65.0 for android',
                                        'chrome 65.0 for ios',
                                        'chrome 66.0',
                                        'chrome 66.0 for android',
                                        'chrome 66.0 for ios',
                                        'chrome 67.0',
                                        'chrome 67.0 for android',
                                        'chrome 69.0',
                                        'chrome generic',
                                        'chrome generic for android',
                                        ]),'id_31'] = 'chrome'

    df_train.loc[df_train['id_31'].isin(['mobile safari 11.0', 
                                        'mobile safari 10.0',
                                        'mobile safari 9.0',
                                        'mobile safari 8.0',
                                        'mobile safari uiwebview',
                                        'mobile safari generic',
                                        'safari 10.0',
                                        'safari 11.0',
                                        'safari 9.0',
                                        'safari generic',                                    
                                        ]),'id_31'] = 'safari'

    df_train.loc[df_train['id_31'].isin(['firefox 47.0',
                                        'firefox 48.0', 
                                        'firefox 52.0',
                                        'firefox 55.0',
                                        'firefox 56.0',
                                        'firefox 57.0',
                                        'firefox 58.0',
                                        'firefox 59.0',
                                        'firefox 60.0',
                                        'firefox generic',
                                        'Mozilla/Firefox',
                                        'firefox mobile 61.0',
                                        ]),'id_31'] = 'firefox'

    df_train.loc[df_train['id_31'].isin(['ie 11.0 for desktop', 'ie 11.0 for tablet']),'id_31'] = 'ie'

    df_train.loc[df_train['id_31'].isin(['edge 17.0',
                                        'edge 16.0',
                                        'edge 15.0',
                                        'edge 14.0',
                                        'edge 13.0',
                                        ]),'id_31'] = 'edge'

    df_train.loc[df_train['id_31'].isin(['opera 49.0',
                                        'opera 51.0',
                                        'opera 52.0',
                                        'opera 53.0',
                                        'opera generic',    
                                        ]),'id_31'] = 'opera'

    df_train.loc[df_train['id_31'].isin(['samsung browser 3.3',
                                        'samsung browser 4.0',
                                        'samsung browser 4.2',
                                        'samsung browser 5.2',
                                        'samsung browser 5.4',
                                        'samsung browser 6.2',
                                        'samsung browser 6.4',
                                        'samsung browser 7.0',
                                        'samsung browser generic',
                                        'Samsung/SM-G531H',
                                        'Samsung/SM-G532M',
                                        'Samsung/SCH',
                                        ]),'id_31'] = 'samsung'

    df_train.loc[df_train['id_31'].isin([
                                        'icedragon',
                                        'comodo',
                                        'mobile',
                                        'google',
                                        'ZTE/Blade',
                                        'Lanix/Ilium',
                                        'android webview 4.0',
                                        'Generic/Android 7.0',
                                        'android browser 4.0',
                                        'Generic/Android',
                                        'google search application 48.0',
                                        'google search application 49.0',
                                        'Microsoft/Windows',
                                        'silk',
                                        'line',                                  
                                        'maxthon',                               
                                        'aol',                                   
                                        'palemoon',                              
                                        'puffin',                                
                                        'facebook',                              
                                        'waterfox',                              
                                        'Cherry',     
                                        'android',   
                                        'Inco/Minion',
                                        'cyberfox',   
                                        'chromium',   
                                        'M4Tel/M4',   
                                        'Nokia/Lumia',
                                        'seamonkey',  
                                        'BLU/Dash',   
                                        'iron',       
                                        'LG/K-200',
                                        ]),'id_31'] = 'other'


    # DeviceInfo
    df_train['DeviceInfo'] = df_train['DeviceInfo'].str.lower()

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('alcatel') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('4013') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('4047') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('501') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('5020') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('5025') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('5049') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('5054') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('5056') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('5080') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('5085') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('5049w') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('9003a') == True] = 'ALCATEL'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('9008a') == True] = 'ALCATEL'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('android') == True] = 'Android'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('blade') == True] = 'BLADE'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('hisense') == True] = 'HISENSE'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('htc') == True] = 'HTC'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('huawei') == True] = 'HUAWEI'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('/huawei') == True] = 'HUAWEI'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('hi6210sft') == True] = 'HUAWEI'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('ilium') == True] = 'ILIUM'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('ios device') == True] = 'iOS'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('lenovo') == True] = 'LENOVO'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('mot-') == True] = 'LENOVO'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('lg') == True] = 'LG'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('lg-') == True] = 'LG'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('vs') == True] = 'LG'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('linux') == True] = 'LINUX'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('macos') == True] = 'Mac OS'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('moto') == True] = 'MOTO'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('mot') == True] = 'MOTO'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('xt') == True] = 'MOTO'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('m4') == True] = 'M4'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('nexus') == True] = 'NEXUS'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('ta-') == True] = 'NOKIA'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('nokia') == True] = 'NOKIA'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('pixel') == True] = 'PIXEL'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('redmi') == True] = 'REDMI'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('rv:') == True] = 'rv:'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('samsung') == True] = 'SAMSUNG'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('sm-') == True] = 'SAMSUNG'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('gt-') == True] = 'SAMSUNG'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('gxq6580_weg_l') == True] = 'SAMSUNG'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('sgh-') == True] = 'SAMSUNG'
        
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('c6906') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('d5316') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('d5803') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('d6603') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('g8141') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('f8331') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('sov33') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('e2') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('e5') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('e6') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('f3') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('f5') == True] = 'SONY'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('g3') == True] = 'SONY'    
        
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('windows') == True] = 'Windows'

    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('z8') == True] = 'ZTE'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('z9') == True] = 'ZTE'
    df_train['DeviceInfo'][df_train['DeviceInfo'].str.contains('zte') == True] = 'ZTE'   
        
    df_train.loc[~df_train['DeviceInfo'].isin([
                                            'Windows'
                                            'iOS'
                                            'Mac OS',
                                            'SAMSUNG',
                                            'MOTO',
                                            'LG',
                                            'HUAWEI',
                                            'SONY',
                                            'ALCATEL',
                                            'BLADE',
                                            'HTC',
                                            'LENOVO',
                                            'ILIUM',
                                            'PIXEL',
                                            'HISENSE',
                                            'M4',
                                            'REDMI',
                                            'ZTE',                               
                                            'NOKIA',
                                            'LINUX',
                                            'NEXUS',
                                            'Android',
                                        ]),'DeviceInfo'] = 'Other'


    # P_emaildomain
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('gmail') == True] = 'gmail'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('yahoo') == True] = 'yahoo'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('outlook') == True] = 'outlook'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('hotmail') == True] = 'outlook'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('msn') == True] = 'outlook'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('live') == True] = 'outlook'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('anonymous') == True] = 'anonymous'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('aol') == True] = 'aol'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('icloud') == True] = 'icloud'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('mac') == True] = 'icloud'
    df_train['P_emaildomain'][df_train['P_emaildomain'].str.contains('me') == True] = 'icloud'
    df_train.loc[~df_train['P_emaildomain'].isin([
                                        'gmail',
                                        'yahoo',
                                        'outlook',
                                        'anonymous',
                                        'aol',
                                        'icloud',
                                        ]),'P_emaildomain'] = 'other'


    # R_emaildomain
    df_train.R_emaildomain.loc[df_train.R_emaildomain.str.contains('gmail') == True] = 'gmail'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('yahoo') == True] = 'yahoo'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('outlook') == True] = 'outlook'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('hotmail') == True] = 'outlook'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('msn') == True] = 'outlook'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('live') == True] = 'outlook'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('anonymous') == True] = 'anonymous'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('aol') == True] = 'aol'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('icloud') == True] = 'icloud'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('mac') == True] = 'icloud'
    df_train['R_emaildomain'][df_train['R_emaildomain'].str.contains('me.') == True] = 'icloud'
    df_train.loc[~df_train['R_emaildomain'].isin([
                                        'gmail',
                                        'yahoo',
                                        'anonymous',
                                        'aol',
                                        'outlook',
                                        'icloud',
                                        ]),'R_emaildomain'] = 'other'

    return df_train