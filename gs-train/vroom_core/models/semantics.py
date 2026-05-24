import torch
import torch.nn.functional as F

class SemanticsManager:
    def __init__(self, label_ids):
        self.num_classes = len(label_ids)
        self.label_ids, _ = torch.sort(label_ids.view(-1))

    def build_lookup_table(self, labels):
        """
        map each unique label to an index for one hot encoding
        """
        self.label_ids = self.label_ids.to(labels.device)
        labels_indices = torch.bucketize(labels, self.label_ids)
        return labels_indices

    def one_hot_encode(self, labels_indices):
        return F.one_hot(labels_indices, num_classes=self.num_classes)

    def one_hot_decode(self, one_hot, num_classes):
        indices = torch.argmax(one_hot, dim=1)
        self.label_ids = self.label_ids.to(one_hot.device)
        return self.label_ids[indices]

    def update_current_num_classes(self, labels):
        self.num_classes = len(torch.unique(labels))
