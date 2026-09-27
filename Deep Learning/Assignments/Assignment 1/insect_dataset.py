#this is made with reference to the tutorial https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image

#custom class here is mostly identical to the tutorial, bar a few small changes.
class InsectDataset(Dataset):
    #init method is the same as tutorial.
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    #len method is the same.
    def __len__(self):
        return len(self.img_labels)
    
    #getitem method is almost the same, the difference being that we hardcode the columns wanted from the .csv file.
    def __getitem__(self, idx):
        img_filename = self.img_labels.iloc[idx, 2] #looking at the .csv can see that column 2 is the image filename
        species_label = self.img_labels.iloc[idx, 1] # here can see that column 1 is the species label
        
        img_path = os.path.join(self.img_dir, img_filename)
        image = decode_image(img_path) 
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            species_label = self.target_transform(species_label)
        
        return image, species_label
    
# Here i have a print statement to ensure that its given just like how it says in the assignment example description. Minor sanity check. This was placed at the end of dataset_tester.py
# for i in range(5):
#     image, label = dataset[i]
#     print(f"<image data from {dataset.img_labels.iloc[i, 2]}>,  {label}")