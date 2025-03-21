import torchvision.datasets.voc as voc

class PascalVOC_Dataset(voc.VOCDetection):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        
        super().__init__(
             root, 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)
    
    
    def __getitem__(self, index):
        return super().__getitem__(index)
        
    
    def __len__(self):
        return len(self.images)
