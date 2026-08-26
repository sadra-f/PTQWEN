
class PT:
    def __init__(self, id, batch_index, cue_index=None, def_end_index=None, ref_seq_start_index=None, ref_seq_end_index=None):
        self.id = id
        self.batch_index = batch_index
        self.cue_index = cue_index
        self.def_end_index = def_end_index
        self._ref_seq_start_index = ref_seq_start_index
        self._ref_seq_end_index = ref_seq_end_index
        self.use_indecies = set()
        self.representation = None
    
    @property
    def ref_seq_start_index(self):
        return self._ref_seq_start_index
        
    @ref_seq_start_index.setter
    def ref_seq_start_index_setter(self, value):
        assert value == self.cue_index + 1
        self._ref_seq_start_index = value
    
    @property
    def ref_seq_end_index(self):
        return self._ref_seq_end_index
    
    @ref_seq_end_index.setter
    def ref_seq_end_index_setter(self, value):
        assert value == self.def_end_index - 1
        self._ref_seq_end_index = value

    @property
    def has_representation(self):
        return self.representation is not None

    def __eq__(self, value):
        if isinstance(value, PT):
            return self.id == value.id and \
            self.batch_index == value.batch_index
        return False
    
    def __str__(self):
        return f"pt{self.id}({self.cue_index},{self.def_end_index}): {self.use_indecies}"

    def __repr__(self):
        return str(self)
