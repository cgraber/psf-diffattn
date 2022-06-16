import torch
from typing import Union

from detectron2.structures import Boxes, Instances
from scipy.optimize import linear_sum_assignment


class BatchedInstanceSequence():
    def __init__(self, instances):
        self._batched_cache = {}
        self._batch_size = len(instances)
        self._seq_len = len(instances[0])
        self._image_size = instances[0][0].image_size
        self._instances = instances
        self._fields = self._instances[0][0].get_fields().keys()
        self._num_instances = [[len(b) for b in p] for p in instances]
        self._ninst = [sum(x) for x in self._num_instances]
        self._max_len = max(self._ninst)
        for k,v in self._instances[0][0].get_fields().items():
            if hasattr(v, "device"):
                self._device = v.device
                break

    def __getattr__(self, name: str):
        if name.startswith('_'):
            return super().__getattr__(self, name)
        if name == "_fields"  or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the batched instances!".format(name))
        if name == 'pred_boxes':
            field_size = torch.Size([4])
            device = self._instances[0][0].get(name).tensor.device
            dtype = self._instances[0][0].get(name).tensor.dtype
        else:
            field_size = self._instances[0][0].get(name).shape[1:]
            device = self._instances[0][0].get(name).device
            dtype = self._instances[0][0].get(name).dtype
        final_shape = torch.Size([self._batch_size, self._max_len]) + field_size
        result = torch.zeros(final_shape, device=device, dtype=dtype)
        for b_idx in range(self._batch_size):
            n_instances = self._ninst[b_idx]
            if name == 'pred_boxes':
                items = [self._instances[b_idx][t].get(name).tensor for t in range(self._seq_len)]
            else:
                items = [self._instances[b_idx][t].get(name) for t in range(self._seq_len)]

            result[b_idx, :n_instances] = torch.cat(items)
        return result

    def add_batched_entry(self, name, entry):
        for b_idx in range(self._batch_size):
            num_instances = self._num_instances[b_idx]
            current_idx = 0
            for t_idx in range(self._seq_len):
                c_ninst = num_instances[t_idx]
                current_entry = entry[b_idx, current_idx:current_idx+c_ninst]
                if 'boxes' in name:
                    current_entry = Boxes(current_entry)
                current_idx += c_ninst
                self._instances[b_idx][t_idx].set(name, current_entry)

    def get_mask(self):
        result = torch.zeros(self._batch_size, self._max_len, dtype=torch.bool, device=self._device)
        for b_idx in range(self._batch_size):
            n_instances = self._ninst[b_idx]
            result[:n_instances] = True
        return result


    def get_instances(self):
        return self._instances

    def get_time(self, offset=0):
        result = torch.zeros(self._batch_size, self._max_len, dtype=torch.long, device=self._device)
        for b_idx in range(self._batch_size):
            current_idx = 0
            for t_idx in range(self._seq_len):
                current_len = self._num_instances[b_idx][t_idx]
                result[b_idx, current_idx:current_idx+current_len] = t_idx + offset
                current_idx += current_len
        return result

    def prepare_odom(self, odom, odom_size):
        result = torch.zeros(self._batch_size, self._max_len, odom_size, device=self._device)
        for b_idx in range(self._batch_size):
            current_odom = odom[b_idx].split(1)
            n_inst = self._ninst[b_idx]
            n_instances = self._num_instances[b_idx]
            current_odom = [o.expand(n, -1) for n,o in zip(n_instances, current_odom)]
            result[b_idx, :n_inst] = torch.cat(current_odom)
        return result

                
    def has(self, name: str):
        return name in self._fields        

    def __len__(self):
        return self._seq_len

    def __getitem__(self, item: Union[int, slice]):
        """
        Args:
        item: an index-like object and will be used to index
            this sequencce temporally.
        
        Returns:
            a new BatchedInstanceSequence containing the temporal subset of instances
        """
        if type(item) == int:
            new_instances = [[self._instances[idx][item]] for idx in range(self._batch_size)]
        else:
            new_instances = [self._instances[idx][item] for idx in range(self._batch_size)]
        return BatchedInstanceSequence(new_instances)


    def clone(self, fields=None):
        if fields is None:
            fields = self._fields
        result = []
        for b_idx in range(self._batch_size):
            b_result = []
            result.append(b_result)
            for t_idx in range(self._seq_len):
                new_inst = Instances(self._image_size)
                old_inst = self._instances[b_idx][t_idx]
                for field in fields:
                    try:
                        new_inst.set(field, old_inst.get(field))
                    except:
                        pass
                b_result.append(new_inst)
        return BatchedInstanceSequence(result)

    @staticmethod
    def time_cat(instance_list):
        final_instances = []
        for b_idx in range(instance_list[0]._batch_size):
            final_instances.append(sum([i_l._instances[b_idx] for i_l in instance_list], []))
        return BatchedInstanceSequence(final_instances)