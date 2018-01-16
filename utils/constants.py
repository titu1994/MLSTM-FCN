
TRAIN_FILES = ['../data/arabic/', # 0
               '../data/CK/', # 1
               '../data/character/', # 2
               '../data/Action3D/', # 3
               '../data/Activity/', # 4
               '../data/arabic_voice/', # 5
               '../data/JapaneseVowels/', # 6

               # New benchmark datasets
               '../data/AREM/', # 7
               '../data/gesture_phase/', # 8
               '../data/HT_Sensor/', # 9
               '../data/MovementAAL/', # 10
               '../data/HAR/', # 11
               '../data/occupancy_detect/', # 12
               '../data/eeg/',  # 13
               '../data/ozone/', # 14
               '../data/daily_sport/',  # 15
               '../data/eeg2/',  # 16
               '../data/MHEALTH/',  # 17
               '../data/EEG_Comp2_data1a/',  # 18
               '../data/EEG_Comp2_data1b/',  # 19
               '../data/EEG_Comp2_data3/',  # 20
               '../data/EEG_Comp2_data4/',  # 21
               '../data/EEG_Comp3_data2/',  # 22
               '../data/EEG_Comp3_data1/',  # 23
               '../data/uwave/',  # 24
               '../data/opportunity/',  # 25
               '../data/pamap2/',  # 26
               '../data/WEASEL_MUSE_DATASETS/ArabicDigits/',  # 27
               '../data/WEASEL_MUSE_DATASETS/AUSLAN/',  # 28
               '../data/WEASEL_MUSE_DATASETS/CharacterTrajectories/',  # 29
               '../data/WEASEL_MUSE_DATASETS/CMUsubject16/',  # 30
               '../data/WEASEL_MUSE_DATASETS/ECG/',  # 31
               '../data/WEASEL_MUSE_DATASETS/JapaneseVowels/',  # 32
               '../data/WEASEL_MUSE_DATASETS/KickvsPunch/',  # 33
               '../data/WEASEL_MUSE_DATASETS/Libras/',  # 34
               '../data/WEASEL_MUSE_DATASETS/NetFlow/',  # 35
               '../data/WEASEL_MUSE_DATASETS/PEMS/',  # 36
               '../data/WEASEL_MUSE_DATASETS/UWave/',  # 37
               '../data/WEASEL_MUSE_DATASETS/Wafer/',  # 38
               '../data/WEASEL_MUSE_DATASETS/WalkvsRun/',  # 39
               '../data/WEASEL_MUSE_DATASETS/digitshape_random/',  # 40
               '../data/WEASEL_MUSE_DATASETS/lp1/',  # 41
               '../data/WEASEL_MUSE_DATASETS/lp2/',  # 42
               '../data/WEASEL_MUSE_DATASETS/lp3/',  # 43
               '../data/WEASEL_MUSE_DATASETS/lp4/',  # 44
               '../data/WEASEL_MUSE_DATASETS/lp5/',  # 45
               '../data/WEASEL_MUSE_DATASETS/pendigits/',  # 46
               '../data/WEASEL_MUSE_DATASETS/shapes_random/',  # 47
               ]

TEST_FILES = ['../data/arabic/', # 0
              '../data/CK/', # 1
              '../data/character/', # 2
              '../data/Action3D/', # 3
              '../data/Activity/', # 4
              '../data/arabic_voice/', # 5
              '../data/JapaneseVowels/', # 6

              # New benchmark datasets
              '../data/AREM/', # 7
              '../data/gesture_phase/', # 8
              '../data/HT_Sensor/',  # 9
              '../data/MovementAAL/',  # 10
              '../data/HAR/',  # 11
              '../data/occupancy_detect/',  # 12
              '../data/eeg/', # 13
              '../data/ozone/',  # 14
              '../data/daily_sport/',  # 15
              '../data/eeg2/',  # 16
              '../data/MHEALTH/',  # 17
              '../data/EEG_Comp2_data1a/',  # 18
              '../data/EEG_Comp2_data1b/',  # 19
              '../data/EEG_Comp2_data3/',  # 20
              '../data/EEG_Comp2_data4/',  # 21
              '../data/EEG_Comp3_data2/',  # 22
              '../data/EEG_Comp3_data1/',  # 23
              '../data/uwave/',  # 24
              '../data/opportunity/',  # 25
              '../data/pamap2/',  # 26
              '../data/WEASEL_MUSE_DATASETS/ArabicDigits/',  # 27
              '../data/WEASEL_MUSE_DATASETS/AUSLAN/',  # 28
              '../data/WEASEL_MUSE_DATASETS/CharacterTrajectories/',  # 29
              '../data/WEASEL_MUSE_DATASETS/CMUsubject16/',  # 30
              '../data/WEASEL_MUSE_DATASETS/ECG/',  # 31
              '../data/WEASEL_MUSE_DATASETS/JapaneseVowels/',  # 32
              '../data/WEASEL_MUSE_DATASETS/KickvsPunch/',  # 33
              '../data/WEASEL_MUSE_DATASETS/Libras/',  # 34
              '../data/WEASEL_MUSE_DATASETS/NetFlow/',  # 35
              '../data/WEASEL_MUSE_DATASETS/PEMS/',  # 36
              '../data/WEASEL_MUSE_DATASETS/UWave/',  # 37
              '../data/WEASEL_MUSE_DATASETS/Wafer/',  # 38
              '../data/WEASEL_MUSE_DATASETS/WalkvsRun/',  # 39
              '../data/WEASEL_MUSE_DATASETS/digitshape_random/',  # 40
              '../data/WEASEL_MUSE_DATASETS/lp1/',  # 41
              '../data/WEASEL_MUSE_DATASETS/lp2/',  # 42
              '../data/WEASEL_MUSE_DATASETS/lp3/',  # 43
              '../data/WEASEL_MUSE_DATASETS/lp4/',  # 44
              '../data/WEASEL_MUSE_DATASETS/lp5/',  # 45
              '../data/WEASEL_MUSE_DATASETS/pendigits/',  # 46
              '../data/WEASEL_MUSE_DATASETS/shapes_random/',  # 47

              ]

MAX_NB_VARIABLES = [13,  # 0
                    136,  # 1
                    30,  # 2
                    570,  # 3
                    570,  # 4
                    39,  # 5
                    12,  # 6

                    # New benchmark datasets
                    7,  # 7
                    18,  # 8
                    11,  # 9
                    4,  # 10
                    9,  # 11
                    5,  # 12
                    13,  # 13
                    72,  # 14
                    45,  #15
                    64,  #16
                    23,  #17
                    6,  #18
                    7,  #19
                    3,  #20
                    28,  #21
                    64,  #22
                    64,  #23
                    3,  #24
                    77,  #25
                    52,  #26
                    13,  #27
                    22,  #28
                    3,  #29
                    62,  #30
                    2,  #31
                    12,  #32
                    62,  #33
                    2,  #34
                    4,  #35
                    963,  #36
                    3,  #37
                    6,  #38
                    62,  #39
                    2,  #40
                    6,  #41
                    6,  # 42
                    6,  # 43
                    6,  # 44
                    6,  # 45
                    2, #46
                    2, #46

                    ]

MAX_TIMESTEPS_LIST = [93,  # 0
                      71,  # 1
                      173,  # 2
                      100, # 3
                      337, # 4
                      91, # 5
                      26, # 6

                      # New benchmark datasets
                      480, # 7
                      214, # 8
                      5396, # 9
                      119, # 10
                      128, # 11
                      3758, # 12
                      117, # 13
                      291, # 14
                      125, #15
                      256, #16
                      42701, #17
                      896, #18
                      1152, #19
                      1152, #20
                      500,#21
                      7794, #22
                      3000, #23
                      315, #24
                      24, #25
                      34, #26
                      93, #27
                      96, #28
                      205, #29
                      534, #30
                      147, #31
                      26, #32
                      761, #33
                      45, #34
                      994, #35
                      144, #36
                      315, #37
                      198, #38
                      1918, #39
                      97, #40
                      15, #41
                      15, #42
                      15, #43
                      15, #44
                      15, #45
                      8, #46
                      97, #47

                      ]


NB_CLASSES_LIST = [10, # 0
                   7, # 1
                   20, # 2
                   20, # 3
                   16, # 4
                   88, # 5
                   9, # 6

                   # New benchmark datasets
                   7, # 7
                   5, # 8
                   3, # 9
                   2, # 10
                   6, # 11
                   2, # 12
                   2, # 13
                   2, # 14
                   19, #15
                   2, #16
                   13, #17
                   2, #18
                   2, #19
                   2, #20
                   2,#21
                   29, #22
                   2, #23
                   8, #24
                   18, #25
                   12, #26
                   10, #27
                   95, #28
                   20, #29
                   2, #30
                   2, #31
                   9, #32
                   2, #33
                   15, #34
                   2, #35
                   7, #36
                   8, #37
                   2, #38
                   2, #39
                   4, #40
                   4, #41
                   5, #42
                   4, #43
                   3, #44
                   5, #45
                   10, #46
                   3,#47

                   ]