from typing import Literal, Iterable
import  torch

class PT:
    def __init__(self, id, batch_index, cue_index=None, def_end_index=None, ref_seq_start_index=None, ref_seq_end_index=None):
        self.id = id
        self.batch_index = batch_index
        self.cue_index = cue_index
        self.def_end_index = def_end_index
        self._ref_seq_start_index = ref_seq_start_index
        self._ref_seq_end_index = ref_seq_end_index
        self.use_indecies = set()
    
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

    def __eq__(self, value):
        if isinstance(value, PT):
            return self.id == value.id and \
            self.batch_index == value.batch_index
        return False
    
    def __str__(self):
        return f"pt{self.id}({self.cue_index},{self.def_end_index}): {self.use_indecies}"

    def __repr__(self):
        return str(self)

class ManagePT:
    def __init__(self, cue_id, pt_ids:Iterable, redist_multiplier=0.85):
        self.all_pt_ids = list([cue_id])
        self.all_pt_ids.extend(pt_ids)
        self.all_pt_ids = sorted(self.all_pt_ids)
        self.cue_id = cue_id
        self.pt_ids = pt_ids
        self.pts = None
        self.redist_multiplier = redist_multiplier
        self.last_seen_index = -1
        self.prev_id = None
        self.prev_ri = None

    def _process_single_new_token(self, batched_seq): #TODO: don't redefine a token that is defined previously
        assert batched_seq.shape[1] == 1, f"Use this method in case of the LLM in incermenetal decoding mode! input batched seq has the sequence length {batched_seq.shape[1]} !"
        curr_index = self.last_seen_index + 1
        for bi, batch in enumerate(batched_seq):
            curr_id = int(batch[-1])
            if curr_id in self.pt_ids:
                if self._pt_exist(bi, curr_id):
                    if self.prev_id[bi] == self.cue_id:
                        print(f"Definition for PT {curr_id} Seen at batch {bi} and index {curr_index} But was previously defined. Skipping addition.")
                    else:
                        self.pts[bi][int(curr_id)].use_indecies.add(curr_index)
                elif self.prev_id[bi] == self.cue_id:
                    self._add_pt(bi, curr_id, cue_index=self.prev_ri[bi], def_end_index=curr_index)
                else:
                    print(f"PT {curr_id} Seen at batch {bi} and index {curr_index} But not defined previously. Skipping addition.")
            elif curr_id == self.cue_id:
                pass
            if curr_id in self.all_pt_ids:
                self.prev_ri[bi] = curr_index
                self.prev_id[bi] = curr_id
        
    def extract_PTs(self, batched_seq, is_training=True):
        """Traverses the sequence for each batch and extracts the PTs and their associated information.
            

        Args:
            batched_seq (torch.Tensor): A 2d tensor of batches of input ids to the LLM.
            mode (Literal[&quot;training&quot;,&quot;eval&quot;], optional): If model in eval mode only latest token is checked and not all the sequence. Defaults to 'training'.
        """
        assert len(batched_seq.shape) == 2, f"Expected a 2d tensor for batched_seq(batch_count, seq_length), but got shape {batched_seq.shape}."
        if not is_training:
            raise NotImplementedError("Sorry!")
        if batched_seq.shape[1] == 1:
            self._process_single_new_token(batched_seq)
            self.last_seen_index += 1
            return
        self.prev_id = [-1 for _ in range(batched_seq.shape[0])]
        self.prev_ri = [-1 for _ in range(batched_seq.shape[0])]
        self.pts = [{} for _ in range(batched_seq.shape[0])]
        self.last_seen_index = -1
        # Pick (batch_index, seq_index) of indecies within the PT tokens range min >= & <= max
        batched_related = torch.where(((batched_seq >= self.all_pt_ids[0]) & (batched_seq <= self.all_pt_ids[-1])))

        # bi => batch_index , ri => related_index (of sequence ids within the batch: bi)
        for bi, ri in zip(batched_related[0], batched_related[1]):
            curr_id = int(batched_seq[bi,ri])
            if curr_id in self.pt_ids:
                if self._pt_exist(bi, curr_id):
                    self.pts[bi][curr_id].use_indecies.add(ri)
                elif self.prev_id[bi] == self.cue_id:
                    self._add_pt(bi, curr_id, cue_index=self.prev_ri[bi], def_end_index=ri)
                else:
                    print(f"PT {curr_id} Seen at batch {bi} and index {ri} But not defined previously. Skipping addition.")
            elif curr_id == self.cue_id:
                # nothing comes to mind that requires action when a cue is seen initialy, 
                # but we can add logic here if needed in the future.
                pass 
            self.prev_ri[bi] = ri
            self.prev_id[bi] = curr_id
            
        self.last_seen_index += batched_seq.shape[1] # !!CAREFUL!!: if code gets here more than without resetting the state lists (self.prev..) the difference between index and length will casue trouble!
    
    def _pt_exist(self, batch_index, id:int):
        return id in self.pts[batch_index]

    def _add_pt(self, batch_index, id:int, cue_index=None, def_end_index=None):
        assert not self._pt_exist(batch_index, id), f"PT with id {id} already exists in batch {batch_index}. Cannot add duplicate PTs."
        self.pts[batch_index][id] = PT(id, batch_index, cue_index, def_end_index, cue_index+1, def_end_index-1)

    def calculate_bias(self, batched_att_weights):
        """Calculates the bias for each PT in each batch based on the provided tensors and Previosuly Found PT references.

        Args:
            batched_tensors (torch.Tensor): A 4d tensor of attention weights [batch_count, num_heads, seq_length, seq_length].
        """
        assert self.pts is not None, "PTs have not been extracted yet. Please call extract_PTs first."
        res_bias = torch.zeros_like(batched_att_weights, device=batched_att_weights.device)  # Placeholder for actual bias calculation logic
        multiplier = torch.ones_like(batched_att_weights)
        self.pts:list[dict[int, PT]]
        # start from end to start of sequence(i.e. [,,,-1]) for each existing pt use divide attention given to 
        # it between itself (0.1), previous same pts(0.1/n) and referenced sequence (0.8)
        # bpts => batch_PTs
        for bpts in self.pts:
            for pt_tkn_id, pto in bpts.items():
                for seq_index in pto.use_indecies:
                    att_weights = batched_att_weights[pto.batch_index, :, :, seq_index]
                    res_bias[pto.batch_index, :, :, seq_index] = att_weights * -self.redist_multiplier
                    res_bias[pto.batch_index, :, :, pto.ref_seq_start_index:pto.def_end_index] = (att_weights * self.redist_multiplier).unsqueeze(-1)
        return res_bias
