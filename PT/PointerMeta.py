
class PT:
    def __init__(self, id, batch_index, cue_index=None, def_end_index=None, ref_seq_start_index=None, ref_seq_end_index=None):
        self.id = id
        self.batch_index = batch_index
        self.cue_index = cue_index
        self.def_end_index = def_end_index
        self._ref_seq_start_index = ref_seq_start_index
        self._ref_seq_end_index = ref_seq_end_index
        self._use_indices = set()
        self.representation = None
        self._req_hidden_states = dict()
    
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

    @property
    def entire_def_indices(self):
        return [i for i in range(self.cue_index, self.def_end_index+1, 1)]

    @property
    def def_and_use_indices(self):
        return self.entire_def_indices + list(self._use_indices)

    @property
    def is_used(self):
        return len(self._use_indices) > 0

    @property
    def use_indices(self):
        return list(self._use_indices)

    def get_req_hidden_states(self, layer_index):
        if layer_index in self._req_hidden_states.keys():
            return self._req_hidden_states[layer_index]
    
    def set_req_hidden_states(self, layer_index, value):
        self._req_hidden_states[layer_index] = value

    def reset_req_hidden_states(self, layer_index):
        self._req_hidden_states[layer_index] = None

    def __eq__(self, value):
        if isinstance(value, PT):
            return self.id == value.id and \
            self.batch_index == value.batch_index
        return False
    
    def __str__(self):
        return f"pt{self.id}({self.cue_index},{self.def_end_index}): {self._use_indices}"

    def __repr__(self):
        return str(self)

