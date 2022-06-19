# Reducing the number of the features

import pandas as pd
import numpy as np

def classreduce(train, test):

    for df_train in [train, test]:
        # id_30 
        df_train['id_30'][df_train['id_30'].str.contains('Window') == True] = 'Windows'
        df_train['id_30'][df_train['id_30'].str.contains('Android') == True] = 'Android'
        df_train['id_30'][df_train['id_30'].str.contains('iOS') == True] = 'iOS'
        df_train['id_30'][df_train['id_30'].str.contains('Mac') == True] = 'Mac'
        df_train['id_30'][df_train['id_30'].str.contains('Linux') == True] = 'Linux'
        df_train.loc[~df_train['id_30'].isin(['Windows',
                                            'Android',
                                            'iOS',
                                            'Mac',
                                            'Linux',
                                            ]),'id_30'] = 'Other'


        # id_31
        df_train['id_31'][df_train['id_31'].str.contains('chrome') == True] = 'chrome'
        df_train['id_31'][df_train['id_31'].str.contains('safari') == True] = 'safari'
        df_train['id_31'][df_train['id_31'].str.contains('firefox') == True] = 'firefox'
        df_train['id_31'][df_train['id_31'].str.contains('edge') == True] = 'edge'
        df_train['id_31'][df_train['id_31'].str.contains('samsung') == True] = 'samsung'
        df_train['id_31'][df_train['id_31'].str.contains('ie') == True] = 'ie'
        df_train['id_31'][df_train['id_31'].str.contains('opera') == True] = 'opera'
        df_train.loc[~df_train['id_31'].isin(['chrome',
                                            'safari',
                                            'firefox',
                                            'edge',
                                            'samsung',
                                            'ie',
                                            'opera',
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
                                                'Windows',
                                                'iOS',
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

    return train, test