# Used LLM for assistance to generate meanings for each dimension.
labels = [
    "urban heritage",        # Dimension 0
    "natural reserve",       # Dimension 1
    "colonial city",         # Dimension 2
    "gothic monument",       # Dimension 3
    "marine exploration",    # Dimension 4
    "royal estate",          # Dimension 5
    "coastal architecture",  # Dimension 6
    "alpine vista",          # Dimension 7
    "formal gardens",        # Dimension 8
    "sacred art",            # Dimension 9
    "imperial tombs",        # Dimension 10
    "ancient village",       # Dimension 11
    "industrial relics",     # Dimension 12
    "panoramic hike",        # Dimension 13
    "religious complex",     # Dimension 14
    "underground marvel",    # Dimension 15
    "forest sanctuary",      # Dimension 16
    "arabic/middle eastern culture",        # Dimension 17
    "avian fossil",          # Dimension 18
    "burial shrine",         # Dimension 19
    "military stronghold",   # Dimension 20
    "river crossing",        # Dimension 21
    "volcanic mission",      # Dimension 22
    "rocky journey",         # Dimension 23
    "marine fort",           # Dimension 24
    "rural waterways",       # Dimension 25
    "evolution trail",       # Dimension 26
    "canal industry",        # Dimension 27
    "academic reef",         # Dimension 28
    "dynastic mosaic",       # Dimension 29
    "coral habitat",         # Dimension 30
    "guided abbey",          # Dimension 31
    "vineyard trail",        # Dimension 32
    "desert enclave",        # Dimension 33
    "forest bus",            # Dimension 34
    "buddhist heritage",     # Dimension 35
    "tasting ruins",         # Dimension 36
    "shrine route",          # Dimension 37
    "painted valley",        # Dimension 38
    "villa culture",         # Dimension 39
    "colonial mosaic",       # Dimension 40
    "mound system",          # Dimension 41
    "ice street",            # Dimension 42
    "hydraulic trail",       # Dimension 43
    "desert tower",          # Dimension 44
    "mission landscape",     # Dimension 45
    "campus pyramid",        # Dimension 46
    "volcanic mosque",       # Dimension 47
    "empire aqueduct",       # Dimension 48
    "coimbra fresco",        # Dimension 49
    "station route",         # Dimension 50
    "renaissance shrine",    # Dimension 51
    "coal remains",          # Dimension 52
    "caribbean hike",        # Dimension 53
    "capital staff",         # Dimension 54
    "ice square",            # Dimension 55
    "iranian tour",          # Dimension 56
    "endemic species",      # Dimension 57
    "kremlin walk",         # Dimension 58
    "grotto road",          # Dimension 59
    "mayan centre",         # Dimension 60
    "painting town",        # Dimension 61
    "wooden carvings",      # Dimension 62
    "hill landscape",       # Dimension 63
    "elephant mill",        # Dimension 64
    "boat shop",            # Dimension 65
    "preserved building",   # Dimension 66
    "santiago road",        # Dimension 67
    "ticket cliff",         # Dimension 68
    "mountain beach",       # Dimension 69
    "hanseatic fall",       # Dimension 70
    "situated railway",     # Dimension 71
    "hunting station",      # Dimension 72
    "oasis entrance",       # Dimension 73
    "dolmen group",         # Dimension 74
    "black bay",            # Dimension 75
    "room people",          # Dimension 76
    "way trail",            # Dimension 77
    "portuguese horse",     # Dimension 78
    "interesting building", # Dimension 79
    "feature view",         # Dimension 80
    "whale complex",        # Dimension 81
    "baroque convent",      # Dimension 82
    "gold animal",          # Dimension 83
    "family camp",          # Dimension 84
    "african trade",        # Dimension 85
    "urban salt",           # Dimension 86
    "george period",        # Dimension 87
    "exhibition street",    # Dimension 88
    "greek tower",          # Dimension 89
    "rainforest mount",     # Dimension 90
    "lava delta",           # Dimension 91
    "network component",    # Dimension 92
    "islamic australia",    # Dimension 93
    "indian range",         # Dimension 94
    "cemetery el",          # Dimension 95
    "night reef",           # Dimension 96
    "middle art",           # Dimension 97
    "group world",          # Dimension 98
    "information waterfall",# Dimension 99
    "water areas"           # Dimension 100
    "rice terrace",         # 101
    "traditional",  # 102
    "mining fortress",      # 103
    "plant gate",           # 104
    "coastal",       # 105
    "canyon",       # 106
    "architecture",       # 107
    "stone",         # 108
    "agricultural statue",  # 109
    "lushan archaeological",# 110
    "guided abbey",         # 111
    "snow trading",         # 112
    "english silk",         # 113
    "settlement sea",       # 114
    "russian",         # 115
    "dynasty room",         # 116
    "cuenca well",          # 117
    "old architecture", # 118
    "europe train",         # 119
    "university",   # 120
    "colorful",# 121
    "necropolis wetland",   # 122
    "southern",     # 123
    "romanesque",      # 124
    "interesting",     # 125
    "ancient",      # 126
    "urban",      # 127
    "sunset",      # 128
    "sundarbans basilica",  # 129
    "harbour endemic",      # 130
    "inca cruise",          # 131
    "structure rhine",      # 132
    "pre hotel",            # 133
    "giant planning",       # 134
    "worth along",          # 135
    "atlantic column",      # 136
    "trade routes",         # 136
    "urban architecture",   # 137
    "ancient art",          # 138
    "coastal heritage",     # 139
    "wild nature",          # 140
    "cultural walks",       # 141
    "historic spa",         # 142
    "stone sanctuaries",    # 143
    "historic roads",       # 144
    "regional zones",       # 145
    "colonial plateau",     # 146
    "architectural styles", # 147
    "rail transport",       # 148
    "African heritage",     # 149
    "gates & light",        # 150
    "basilica sites",       # 151
    "opera culture",        # 152
    "art carvings",         # 153
    "natural groups",       # 154
    "neolithic view",       # 155
    "pueblo zone",          # 156
    "endemic culture",      # 157
    "university water",     # 158
    "prehistoric site",     # 159
    "largest colony",       # 160
    "networked heritage",   # 161
    "romanesque Asia",      # 162
    "saint structures",     # 163
    "salt region",          # 164
    "fjord civilization",   # 165
    "archaeological sites", # 166
    "medieval influence",   # 167
    "terraced hill",        # 168
    "wild coast",           # 169
    "windmill camp",        # 170
    "agave path",           # 171
    "wooden architecture",  # 172
    "historical port",      # 173
    "atlantic trading",     # 174
    "canal culture",        # 175
    "market culture",       # 176
    "volcanic convent",     # 177
    "ceremonial bath",      # 178
    "citadel time",         # 179
    "engineering spirit",   # 180
    "neolithic wall",       # 181
    "factory gates",        # 182
    "monumental gulf",      # 183
    "coastal style",        # 184
    "holy gate",            # 185
    "port periods",         # 186
    "salt housing",         # 187
    "pre-oasis",            # 188
    "intact views",         # 189
    "delta planning",       # 190
    "pyramid sites",        # 191
    "scenic sites",    # Dimension 192
    "urban travel",    # Dimension 193
    "historic regions",# Dimension 194
    "ancient settlements", # Dimension 195
    "industrial heritage", # Dimension 196
    "architectural mixture", # Dimension 197
    "archaeological sites", # Dimension 198
    "prehistoric legacy" # Dimension 199
]
