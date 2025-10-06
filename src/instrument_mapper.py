"""
乐器名称映射工具
"""

class InstrumentMapper:
    """乐器名称映射类"""
    
    # IRMAS数据集乐器映射
    IRMAS_MAPPING = {
        'cel': 'Cello',
        'cla': 'Clarinet', 
        'flu': 'Flute',
        'gac': 'Acoustic Guitar',
        'gel': 'Electric Guitar',
        'org': 'Organ',
        'pia': 'Piano',
        'sax': 'Saxophone',
        'tru': 'Trumpet',
        'vio': 'Violin',
        'voi': 'Voice'
    }
    
    # 反向映射（英文到缩写）
    REVERSE_MAPPING = {v: k for k, v in IRMAS_MAPPING.items()}
    
    # 乐器类别描述
    INSTRUMENT_DESCRIPTIONS = {
        'Cello': 'Bowed string instrument, bass voice of the violin family',
        'Clarinet': 'Woodwind instrument with a single-reed mouthpiece',
        'Flute': 'Woodwind instrument that produces sound from the flow of air',
        'Acoustic Guitar': 'String instrument that produces sound acoustically',
        'Electric Guitar': 'Guitar that uses pickups to convert string vibration into electrical signals',
        'Organ': 'Keyboard instrument of one or more pipe divisions',
        'Piano': 'Acoustic stringed keyboard instrument',
        'Saxophone': 'Woodwind instrument made of brass',
        'Trumpet': 'Brass instrument commonly used in classical and jazz ensembles',
        'Violin': 'Bowed string instrument, smallest and highest-pitched in the violin family',
        'Voice': 'Human singing voice'
    }
    
    @classmethod
    def get_english_name(cls, abbreviation):
        """获取英文名称"""
        return cls.IRMAS_MAPPING.get(abbreviation, abbreviation)
    
    @classmethod
    def get_abbreviation(cls, english_name):
        """获取缩写"""
        return cls.REVERSE_MAPPING.get(english_name, english_name)
    
    @classmethod
    def get_description(cls, name):
        """获取乐器描述"""
        # 先尝试作为缩写查找
        english_name = cls.get_english_name(name)
        return cls.INSTRUMENT_DESCRIPTIONS.get(english_name, "No description available")
    
    @classmethod
    def translate_labels(cls, labels):
        """翻译标签列表"""
        if isinstance(labels, list):
            return [cls.get_english_name(label) for label in labels]
        elif isinstance(labels, np.ndarray):
            return np.array([cls.get_english_name(label) for label in labels])
        else:
            return labels
    
    @classmethod
    def print_instrument_info(cls):
        """打印所有乐器信息"""
        print("IRMAS Dataset Instrument Information:")
        print("=" * 50)
        for abbrev, english_name in cls.IRMAS_MAPPING.items():
            description = cls.INSTRUMENT_DESCRIPTIONS.get(english_name, "No description")
            print(f"{abbrev:4s} -> {english_name:15s} : {description}")

    @staticmethod
    def get_english_name_static(abbreviation):
        """静态方法获取英文名称"""
        return InstrumentMapper.get_english_name(abbreviation)