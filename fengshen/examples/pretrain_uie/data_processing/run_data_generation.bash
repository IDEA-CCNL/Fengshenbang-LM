#!/usr/bin/env bash
# -*- coding:utf-8 -*-

# for data_format in entity relation event absa
# do
#     python uie_convert.py -format spotasoc -config data_config/${data_format} -output ${data_format}
# done


# ========================== 
# data_format=event_zh
# python uie_convert.py -format spotasoc -config data_config/${data_format} -output ${data_format}

# data_format=entity_zh
# python uie_convert.py -format spotasoc -config data_config/${data_format} -output ${data_format}

# data_format=relation_zh
# python uie_convert.py -format spotasoc -config data_config/${data_format} -output ${data_format}

# ======================== 

# data_format=entity_zh_aligned
# python uie_convert.py -format spotasoc -config data_config/${data_format} -output ${data_format}

# data_format=relation_zh_aligned
# python uie_convert.py -format spotasoc -config data_config/${data_format} -output ${data_format}

# data_format=event_zh_aligned
# python uie_convert.py -format spotasoc -config data_config/${data_format} -output ${data_format}
# =============


# python scripts/data_statistics.py -data converted_data/text2spotasoc/
# ========================== Datasets Statistic OverView ===============================
# name                                                       entity    relation    event    train    val    test    train entity    train relation    train event    train role    val entity    val relation    val event    val role    test entity    test relation    test event    test role
# -------------------------------------------------------  --------  ----------  -------  -------  -----  ------  --------------  ----------------  -------------  ------------  ------------  --------------  -----------  ----------  -------------  ---------------  ------------  -----------
# converted_data/text2spotasoc/entity_zh/DianShang                4           0        0     7998      0       0           25546                 0              0             0             0               0            0           0              0                0             0            0
# converted_data/text2spotasoc/entity_zh/MEDICAL                 13           0        0    55722      0       0           59801                 0              0             0             0               0            0           0              0                0             0            0
# converted_data/text2spotasoc/entity_zh/PeopleDaily              4           0        0    19357      0       0           66593                 0              0             0             0               0            0           0              0                0             0            0
# converted_data/text2spotasoc/entity_zh_aligned/CLUENER         10           0        0    10748   1343    1345           23340                 0              0             0          2982               0            0           0              0                0             0            0
# converted_data/text2spotasoc/entity_zh_aligned/SciCN            6           0        0    10915   1600    3115           44046                 0              0             0          6183               0            0           0          11844                0             0            0
# converted_data/text2spotasoc/entity_zh_aligned/Weibo            8           0        0     1350    270     270            1895                 0              0             0           389               0            0           0            418                0             0            0
# converted_data/text2spotasoc/event_zh_aligned/CDEE              0           0        1     1587    384     514               0                 0           8761          9847             0               0         2138        2443              0                0             0            0
# converted_data/text2spotasoc/event_zh_aligned/DUEE              0           0       65    11908   1492   34904               0                 0          13860         28905             0               0         1783        3682              0                0             0            0
# converted_data/text2spotasoc/event_zh_aligned/DUEE-Fin          0           0       13     7015   1171   59394               0                 0           9498         48891             0               0         1533        7915              0                0             0            0
# converted_data/text2spotasoc/relation_zh/CMIE                  11          44        0    14339   3585       0           87320             43660              0             0         21252           10626            0           0              0                0             0            0
# converted_data/text2spotasoc/relation_zh/DuIE                  26          48        0   171135  20652       0          620756            310378              0             0         75578           37789            0           0              0                0             0            0
# converted_data/text2spotasoc/relation_zh/FinRE                  1          44        0    13486   1489       0           26972             13486              0             0          2978            1489            0           0              0                0             0            0
# converted_data/text2spotasoc/relation_zh/SciCN                  6           4        0     2448    351       0           44046             18795              0             0          6183            2804            0           0              0                0             0            0
# converted_data/text2spotasoc/relation_zh_aligned/CMIE          11          44        0    14339   3585    4482           87320             43660              0             0         21252           10626            0           0              0                0             0            0
# converted_data/text2spotasoc/relation_zh_aligned/DuIE          26          48        0   171135  20652  101239          620756            310378              0             0         75578           37789            0           0              0                0             0            0
# converted_data/text2spotasoc/relation_zh_aligned/FinRE          1          44        0    13486   1489    3727           26972             13486              0             0          2978            1489            0           0           7454             3727             0            0
# converted_data/text2spotasoc/relation_zh_aligned/SanWen         1          10        0    17227   1793    2220           34454             17227              0             0          3586            1793            0           0           4440             2220             0            0
# converted_data/text2spotasoc/relation_zh_aligned/SciCN          6           4        0     2448    351     700           44046             18795              0             0          6183            2804            0           0          11844             5213             0            0
