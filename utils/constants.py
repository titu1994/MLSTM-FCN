
TRAIN_FILES = ['../data/arabic/', # 0
               '../data/CK/', # 1
               '../data/character/', # 2
               '../data/Action3D/', # 3
               '../data/Activity/', # 4
               '../data/arabic_voice/', # 5
               '../data/JapaneseVowels/', # 6
               ]

TEST_FILES = ['../data/arabic/', # 0
              '../data/CK/', # 1
              '../data/character/', # 2
              '../data/Action3D/', # 3
              '../data/Activity/', # 4
              '../data/arabic_voice/', # 5
              '../data/JapaneseVowels/', # 6
              ]

MAX_NB_VARIABLES = [13, # 0
                    136, # 1
                    30, # 2
                    570, # 3
                    570, # 4
                    39, # 5
                    12, # 6
                    ]

MAX_TIMESTEPS_LIST = [93,  # 0
                      71,  # 1
                      173,  # 2
                      100, # 3
                      337, # 4
                      91, # 5
                      26, # 6
                      ]


NB_CLASSES_LIST = [10, # 0
                   7, # 1
                   20, # 2
                   20, # 3
                   16, # 4
                   88, # 5
                   9, # 6
                   ]