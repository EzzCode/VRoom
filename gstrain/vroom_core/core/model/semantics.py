import torch
import torch.nn.functional as F

class SemanticsManager:
    def __init__(self, label_ids):
        # Ensure 0 (which represents unknown or background) is always in label_ids for saftey
        unique_labels = label_ids.view(-1)
        if 0 not in unique_labels:
            unique_labels = torch.cat([torch.tensor([0], dtype=unique_labels.dtype, device=unique_labels.device), unique_labels])
        self.label_ids, _ = torch.sort(unique_labels)
        self.num_classes = len(self.label_ids)

    def build_lookup_table(self, labels):
        """
        map each unique label to an index for one hot encoding
        """
        self.label_ids = self.label_ids.to(labels.device)
        labels_indices = torch.bucketize(labels, self.label_ids)
        
        # use clamping to make sure the incoming label is within the existing labels
        # if it isnt we clamp it
        clamped_indices = torch.clamp(labels_indices, 0, self.label_ids.size(0) - 1)
        # after clamping we check if this label is truly in the correct index
        # if it isnt its false
        mask_match = (self.label_ids[clamped_indices] == labels)
        # set unknown labels to zero
        labels_indices = torch.where(mask_match, labels_indices, torch.zeros_like(labels_indices))
        
        return labels_indices

    def one_hot_encode(self, labels_indices):
        return F.one_hot(labels_indices, num_classes=self.num_classes)

    def one_hot_decode(self, one_hot, num_classes):
        indices = torch.argmax(one_hot, dim=1)
        self.label_ids = self.label_ids.to(one_hot.device)
        return self.label_ids[indices]

    def update_current_num_classes(self, labels):
        self.num_classes = len(torch.unique(labels))

    def instantiate_semantics(
        self,
        semantic_labels,
        visible_anchors_mask,
        negative_opacity_filter,
        gaussians_per_anchor,
    ):
        """
        map visible anchor labels to one hot encodings for visible Gaussians
        """
        visible_labels = semantic_labels[visible_anchors_mask] # size: [num_vis_anchors]
        visible_label_indices = self.build_lookup_table(visible_labels)
        visible_one_hot = self.one_hot_encode(visible_label_indices).float()
        expanded_one_hot = visible_one_hot.unsqueeze(1).expand(-1, gaussians_per_anchor, -1) # keep number of anchors and number of labels but stretch the middle dim
        return expanded_one_hot[negative_opacity_filter] # size: [num_filtered_gaussians x num_labels]

