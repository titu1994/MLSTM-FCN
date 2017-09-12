
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
              ]

MAX_NB_VARIABLES = [13, # 0
                    136, # 1
                    30, # 2
                    570, # 3
                    570, # 4
                    39, # 5
                    12, # 6

                    # New benchmark datasets
                    7, # 7
                    18, # 8
                    11, # 9
                    4, # 10
                    9, # 11
                    5, # 12
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
                   ]