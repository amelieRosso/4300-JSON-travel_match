# Used LLM for assistance to generate meanings for each dimension.
labels = [
    "urban",
    "nature",
    "historic",
    "medieval",
    "marine",
    "castle",
    "temple",
    "mountains",
    "palace",
    "art",
    "ruins",
    "heritage",
    "industry",
    "hiking",
    "religion",
    "quiet places",
    "mosaic",
    "islamic",
    "old",
    "burial",
    "military",
    "water",
    "volcano",
    "carving",
    "bridge",
    "village",
    "evolution",
    "infrastructure",
    "education",
    "tradition",
    "coral",
    "guided",
    "wine",
    "desert",
    "scenic",
    "buddhist",
    "ancient",
    "japan",
    "railway",
    "culture",
    "colonial",
    "spanish",
    "ice",
    "hydraulic",
    "wildlife",
    "transport",
    "jesuit",
    "mayan",
    "volcanoes",
    "coimbra",
    "travel",
    "heritage",     # 51
    "archaeology",  # 52
    "religion",     # 53
    "civilization", # 54
    "nature",       # 55
    "exploration",  # 56
    "conservation", # 57
    "russia",       # 58
    "settlement",   # 59
    "mesoamerica",  # 60
    "architecture", # 61
    "industry",     # 62
    "ruins",        # 63
    "antiquity",    # 64
    "landscape",    # 65
    "wildlife",     # 66
    "routes",       # 67
    "coastline",    # 68
    "biodiversity", # 69
    "royalty",      # 70
    "transport",    # 71
    "pilgrimage",   # 72
    "silkroad",     # 73
    "ecology",      # 74
    "china",        # 75
    "dynasty",      # 76
    "medieval",     # 77
    "culture",      # 78
    "journey",      # 79
    "colonial",     # 80
    "empires",      # 81
    "gates",        # 82
    "baroque",      # 83
    "families",     # 84
    "villages",     # 85
    "preservation", # 86
    "periods",      # 87
    "streets",      # 88
    "traditions",   # 89
    "rainforest",   # 90
    "geology",      # 91
    "components",   # 92
    "islamic",      # 93
    "coast",        # 94
    "graveyard",    # 95
    "marine",       # 96
    "bengal",       # 97
    "urbanism",     # 98
    "artifacts",    # 99
    "megaliths" ,    # 100
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
    "stone",     # 149
    "aqueduct",        # 150
    "small",       # 151
    "Spain",        # 152
    "art carvings",         # 153
    "natural groups",       # 154
    "neolithic view",       # 155
    "pueblo zone",          # 156
    "endemic culture",      # 157
    "university",     # 158
    "prehistoric site",     # 159
    "largest colony",       # 160
    "heritage",   # 161
    "old",      # 162
    "art",     # 163
    "salt region",          # 164
    "ancient",   # 165
    "archaeological sites", # 166
    "medieval influence",   # 167
    "terraced hill",        # 168
    "wild coast",           # 169
    "religious",        # 170
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
