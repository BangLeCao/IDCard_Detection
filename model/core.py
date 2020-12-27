import os
import pandas as pd
import random
import torch
import torchvision

from model.config import config
from model.utils import default_transforms, xml_to_csv, _is_iterable, read_image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=DataLoader.collate_data, **kwargs)

    def collate_data(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, label_data, image_folder=None, transform=None):
      
        # CSV file : filename, width, height, class, xmin, ymin, xmax, ymax
        if os.path.isfile(label_data):
            self._csv = pd.read_csv(label_data)
        else:
            self._csv = xml_to_csv(label_data)

        if image_folder is None:
            self._root_dir = label_data
        else:
            self._root_dir = image_folder

        if transform is None:
            self.transform = default_transforms()
        else:
            self.transform = transform

    def __len__(self):
        return len(self._csv['image_id'].unique().tolist())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        object_entries = self._csv.loc[self._csv['image_id'] == idx]

        img_name = os.path.join(self._root_dir, object_entries.iloc[0, 0])
        image = read_image(img_name)

        boxes = []
        labels = []
        for object_idx, row in object_entries.iterrows():
            # Read in xmin, ymin, xmax, and ymax
            box = self._csv.iloc[object_idx, 4:8]
            boxes.append(box)
            # Read in the label
            label = self._csv.iloc[object_idx, 3]
            labels.append(label)

        boxes = torch.tensor(boxes).view(-1, 4)

        targets = {'boxes': boxes, 'labels': labels}

        #Augment data
        if self.transform:
            width = object_entries.iloc[0, 1]
            height = object_entries.iloc[0, 2]

            updated_transforms = []
            scale_factor = 1.0
            random_flip = 0.0
            for t in self.transform.transforms:
                updated_transforms.append(t)
                if isinstance(t, transforms.Resize):
                    original_size = min(height, width)
                    scale_factor = original_size / t.size
                elif isinstance(t, transforms.RandomHorizontalFlip):
                    random_flip = t.p

            for t in updated_transforms:
                if isinstance(t, transforms.RandomHorizontalFlip):
                    if random.random() < random_flip:
                        image = transforms.RandomHorizontalFlip(1)(image)
                        for idx, box in enumerate(targets['boxes']):
                            # Flip box's x-coordinates
                            box[0] = width - box[0]
                            box[2] = width - box[2]
                            box[[0,2]] = box[[2,0]]
                            targets['boxes'][idx] = box
                else:
                    image = t(image)

            if scale_factor != 1.0:
                for idx, box in enumerate(targets['boxes']):
                    box = (box / scale_factor).long()
                    targets['boxes'][idx] = box

        return image, targets

class Model:

    def __init__(self, classes=None, device=None):
        self._device = device if device else config['default_device']
        self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        print(self._model)

        if classes:
            in_features = self._model.roi_heads.box_predictor.cls_score.in_features
            self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)
            self._disable_normalize = False
        else:
            classes = config['default_classes']
            self._disable_normalize = True

        self._model.to(self._device)

        self._classes = ['__background__'] + classes
        self._int_mapping = {label: index for index, label in enumerate(self._classes)}

    def _get_predictions(self, images):
        self._model.eval()

        with torch.no_grad():
            if not _is_iterable(images):
                images = [images]

            if not isinstance(images[0], torch.Tensor):
                if self._disable_normalize:
                    defaults = transforms.Compose([transforms.ToTensor()])
                else:
                    defaults = default_transforms()
                images = [defaults(img) for img in images]

            images = [img.to(self._device) for img in images]

            preds = self._model(images)
            preds = [{k: v.to(torch.device('cpu')) for k, v in p.items()} for p in preds]
            return preds

    def predict_object(self, images):

        is_single_image = not _is_iterable(images)
        images = [images] if is_single_image else images
        preds = self._get_predictions(images)

        results = []
        for pred in preds:
            result = ([self._classes[val] for val in pred['labels']], pred['boxes'], pred['scores'])
            results.append(result)

        return results[0] if is_single_image else results

    def fit(self, dataset, val_dataset=None, epochs=10, learning_rate=0.005, momentum=0.9,
            weight_decay=0.0005, gamma=0.1, lr_step_size=3, verbose=True):
        if epochs > 0:
            self._disable_normalize = False

        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, shuffle=True)

        if val_dataset is not None and not isinstance(val_dataset, DataLoader):
            val_dataset = DataLoader(val_dataset)

        losses = []
        losses_traning = []
        parameters = [p for p in self._model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        # Training
        for epoch in range(epochs):
            avg_loss_traning = 0
            if verbose:
                print('Epoch {} of {}'.format(epoch + 1, epochs))

            # Training step
            self._model.train()

            if verbose:
                print('Begin iterating over training dataset')

            iterable = tqdm(dataset, position=0, leave=True) if verbose else dataset
            for images, targets in iterable:
                self._convert_to_int_labels(targets)
                images, targets = self._to_device(images, targets)
                loss_dict = self._model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                avg_loss_traning += total_loss.item()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            avg_loss_traning /= len(dataset.dataset)
            losses_traning.append(avg_loss_traning)
            if verbose:
              print(total_loss)

            # Validation step
            if val_dataset is not None:
                avg_loss = 0
                with torch.no_grad():
                    if verbose:
                        print('Begin iterating over validation dataset')
                    iterable = tqdm(val_dataset, position=0, leave=True) if verbose else val_dataset
                    for images, targets in iterable:
                        self._convert_to_int_labels(targets)
                        images, targets = self._to_device(images, targets)
                        loss_dict = self._model(images, targets)
                        total_loss = sum(loss for loss in loss_dict.values())
                        avg_loss += total_loss.item()

                avg_loss /= len(val_dataset.dataset)
                losses.append(avg_loss)

                if verbose:
                    print('Loss: {}'.format(avg_loss))

            lr_scheduler.step()

        if len(losses) > 0:
            return losses, losses_traning

    def get_internal_model(self):

        return self._model

    def save(self, file):

        torch.save(self._model.state_dict(), file)

    @staticmethod
    def load(file, classes):

        model = Model(classes)
        model._model.load_state_dict(torch.load(file, map_location=model._device))
        return model

    def _convert_to_int_labels(self, targets):
        for idx, target in enumerate(targets):
            labels_array = target['labels']
            labels_int_array = [self._int_mapping[class_name] for class_name in labels_array]
            target['labels'] = torch.tensor(labels_int_array)

    def _to_device(self, images, targets):
        images = [image.to(self._device) for image in images]
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
        return images, targets