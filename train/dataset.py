from torch.utils.data import Dataset, Subset
import os
from PIL import Image


################## PKU Dataset Implementation ##################
class PKUVehicleIdDataset(Dataset):
    """Specific implementation for the PKU VehicleID dataset"""

    def __init__(self, img_dir, train_list_txt, label_offset=0, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.label_offset = label_offset

        self.img_names = []

        # aka the car identities,
        # because PKU image names are arbitrary strings and
        # don't encode the ID like VeRi does, so we need to read them from the text file instead
        self.raw_labels = []

        # Parse the text file
        # each row in train_list_txt is expected to have the format: "img_name label"
        with open(train_list_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1])

                    # Some versions of PKU omit the .jpg in the text file
                    if not img_name.endswith(".jpg"):
                        img_name += ".jpg"

                    self.img_names.append(img_name)
                    self.raw_labels.append(label)

        # Map raw PKU IDs to 0 -> N-1 (To eliminate any gaps in PKU itself)
        unique_raw_ids = sorted(list(set(self.raw_labels)))
        self.id_to_class = {raw_id: i for i, raw_id in enumerate(unique_raw_ids)}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Get the squashed, perfectly contiguous label
        raw_label = self.raw_labels[idx]
        label = self.id_to_class[raw_label]

        if self.transform:
            image = self.transform(image)

        dummy_cam_id = -1  # PKU doesn't have camera IDs, so we return -1 for that field
        return image, label + self.label_offset, dummy_cam_id


################## VeRi Dataset Implementation ##################


class VeRiDataset(Dataset):
    """Specific implementation for VeRi dataset"""

    def __init__(self, img_dir, label_offset=0, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        self.transform = transform
        self.label_offset = label_offset

        # Use IDs from file names and map them to `id_to_class`
        raw_ids = sorted(
            list(set([int(name.split("_")[0]) for name in self.img_names]))
        )
        self.id_to_class = {raw_id: i for i, raw_id in enumerate(raw_ids)}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        parts = img_name.split("_")

        # Get Vehicle ID and map it to the correct class label
        raw_id = int(parts[0])
        label = self.id_to_class[raw_id]

        # Get Camera ID (Strip the 'c' and convert to integer)
        cam_id = int(parts[1].replace("c", ""))

        if self.transform:
            image = self.transform(image)

        return image, label + self.label_offset, cam_id


class VeRiDatasetSubset(Dataset):
    """Helper class to apply transformations to a subset of the dataset"""

    def __init__(self, whole_dataset, subset_indices, transform, label_map=None):
        self.subset = Subset(whole_dataset, subset_indices)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label, cam_id = self.subset[idx]
        if self.transform:
            image = self.transform(image)

        if self.label_map is not None:
            label = self.label_map[label]

        return image, label, cam_id


###################### Generic ReID Test Dataset ##################
class ReIDTestDataset(Dataset):
    """Class to load images from a directory for feature extraction during testing."""

    def __init__(self, directory_path, parse_filename, transform):
        self.directory_path = directory_path
        self.image_names = [f for f in os.listdir(directory_path) if f.endswith(".jpg")]
        self.transform = transform
        self.parse_filename = parse_filename

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = os.path.join(self.directory_path, name)

        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)

        # Parse IDs
        vid, cid = self.parse_filename(name)

        return tensor, vid, cid
